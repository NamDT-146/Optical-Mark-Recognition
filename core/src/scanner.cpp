#include "../include/scanner.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

#include <opencv2/imgproc.hpp>

namespace {

int makeOdd(int value) {
	if (value % 2 == 0) {
		return value + 1;
	}
	return value;
}

cv::Point2f contourCentroid(const std::vector<cv::Point>& contour) {
	const cv::Moments moments = cv::moments(contour);
	if (moments.m00 != 0.0) {
		return cv::Point2f(
				static_cast<float>(moments.m10 / moments.m00),
				static_cast<float>(moments.m01 / moments.m00)
		);
	}

	const cv::Rect box = cv::boundingRect(contour);
	return cv::Point2f(
			static_cast<float>(box.x + (box.width / 2)),
			static_cast<float>(box.y + (box.height / 2))
	);
}

cv::Mat preprocessForAnchors(const cv::Mat& bgr) {
	cv::Mat gray;
	cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

	cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(1.2, cv::Size(8, 8));
	cv::Mat enhanced;
	clahe->apply(gray, enhanced);

	cv::Mat blurred;
	cv::GaussianBlur(enhanced, blurred, cv::Size(3, 3), 0.0);

	const int min_side = std::min(gray.rows, gray.cols);
	int dynamic_block_size = static_cast<int>(min_side * 0.05);
	dynamic_block_size = makeOdd(dynamic_block_size);
	dynamic_block_size = std::max(31, dynamic_block_size);

	cv::Mat thresh;
	cv::adaptiveThreshold(
			blurred,
			thresh,
			255,
			cv::ADAPTIVE_THRESH_GAUSSIAN_C,
			cv::THRESH_BINARY_INV,
			dynamic_block_size,
			3
	);

	const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	cv::morphologyEx(thresh, thresh, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 1);
	return thresh;
}

bool isSquareLikeContour(
		const std::vector<cv::Point>& contour,
		const cv::Size& image_size,
		const omr::AnchorDetectionConfig& config) {
	const double area = cv::contourArea(contour);
	if (area < config.min_area || area > config.max_area) {
		return false;
	}

	const double image_area = static_cast<double>(image_size.width) * static_cast<double>(image_size.height);
	if (area > image_area * 0.05) {
		return false;
	}

	const cv::Rect box = cv::boundingRect(contour);
	if (box.height <= 0 || box.width <= 0) {
		return false;
	}

	const double aspect_ratio = static_cast<double>(box.width) / static_cast<double>(box.height);
	const double min_ratio = 1.0 - config.aspect_ratio_tolerance;
	const double max_ratio = 1.0 + config.aspect_ratio_tolerance;
	if (aspect_ratio < min_ratio || aspect_ratio > max_ratio) {
		return false;
	}

	const std::vector<cv::Point> hull = [&]() {
		std::vector<cv::Point> output;
		cv::convexHull(contour, output);
		return output;
	}();
	const double hull_area = cv::contourArea(hull);
	if (hull_area <= 0.0) {
		return false;
	}

	const double solidity = area / hull_area;
	if (solidity < config.min_solidity) {
		return false;
	}

	const double extent = area / static_cast<double>(box.width * box.height);
	if (extent < config.min_extent) {
		return false;
	}

	const double perimeter = cv::arcLength(contour, true);
	if (perimeter <= 0.0) {
		return false;
	}

	std::vector<cv::Point> approx;
	cv::approxPolyDP(contour, approx, 0.05 * perimeter, true);
	return approx.size() >= 4;
}

std::vector<cv::Point2f> centersFromContours(const std::vector<std::vector<cv::Point>>& contours) {
	std::vector<cv::Point2f> centers;
	centers.reserve(contours.size());
	for (const auto& contour : contours) {
		centers.push_back(contourCentroid(contour));
	}
	return centers;
}

bool inferFourthCorner(const std::vector<cv::Point2f>& points, cv::Point2f* inferred) {
	if (inferred == nullptr || points.size() != 3) {
		return false;
	}

	int a = 0;
	int b = 1;
	double best_d2 = -1.0;
	for (int i = 0; i < 3; ++i) {
		for (int j = i + 1; j < 3; ++j) {
			const cv::Point2f delta = points[i] - points[j];
			const double d2 = static_cast<double>(delta.dot(delta));
			if (d2 > best_d2) {
				best_d2 = d2;
				a = i;
				b = j;
			}
		}
	}

	int c = 0;
	while (c == a || c == b) {
		++c;
	}

	*inferred = points[a] + points[b] - points[c];
	return true;
}

bool rescueFourthContour(
		const std::vector<std::vector<cv::Point>>& all_contours,
		const cv::Point2f& guessed_point,
		int rescue_radius_px,
		std::vector<cv::Point>* rescued_contour,
		cv::Point2f* rescued_center) {
	if (rescued_contour == nullptr || rescued_center == nullptr) {
		return false;
	}

	double min_dist = std::numeric_limits<double>::max();
	bool found = false;

	for (const auto& contour : all_contours) {
		const double area = cv::contourArea(contour);
		if (area < 20.0 || area > 30000.0) {
			continue;
		}

		const cv::Point2f center = contourCentroid(contour);
		const cv::Point2f delta = center - guessed_point;
		const double dist = std::sqrt(static_cast<double>(delta.dot(delta)));
		if (dist < rescue_radius_px && dist < min_dist) {
			min_dist = dist;
			*rescued_contour = contour;
			*rescued_center = center;
			found = true;
		}
	}

	return found;
}

std::vector<cv::Point2f> selectFourExtremes(const std::vector<cv::Point2f>& points) {
	if (points.size() <= 4) {
		return points;
	}

	int idx_tl = 0;
	int idx_br = 0;
	int idx_tr = 0;
	int idx_bl = 0;

	float min_sum = std::numeric_limits<float>::max();
	float max_sum = std::numeric_limits<float>::lowest();
	float min_diff = std::numeric_limits<float>::max();
	float max_diff = std::numeric_limits<float>::lowest();

	for (size_t i = 0; i < points.size(); ++i) {
		const float sum = points[i].x + points[i].y;
		const float diff = points[i].y - points[i].x;

		if (sum < min_sum) {
			min_sum = sum;
			idx_tl = static_cast<int>(i);
		}
		if (sum > max_sum) {
			max_sum = sum;
			idx_br = static_cast<int>(i);
		}
		if (diff < min_diff) {
			min_diff = diff;
			idx_tr = static_cast<int>(i);
		}
		if (diff > max_diff) {
			max_diff = diff;
			idx_bl = static_cast<int>(i);
		}
	}

	return {
		points[idx_tl],
		points[idx_tr],
		points[idx_br],
		points[idx_bl]
	};
}

std::vector<cv::Point2f> sortCornersTlTrBrBl(const std::vector<cv::Point2f>& corners) {
	if (corners.size() != 4) {
		return corners;
	}

	int idx_tl = 0;
	int idx_br = 0;
	int idx_tr = 0;
	int idx_bl = 0;

	float min_sum = std::numeric_limits<float>::max();
	float max_sum = std::numeric_limits<float>::lowest();
	float min_diff = std::numeric_limits<float>::max();
	float max_diff = std::numeric_limits<float>::lowest();

	for (int i = 0; i < 4; ++i) {
		const float sum = corners[i].x + corners[i].y;
		const float diff = corners[i].y - corners[i].x;

		if (sum < min_sum) {
			min_sum = sum;
			idx_tl = i;
		}
		if (sum > max_sum) {
			max_sum = sum;
			idx_br = i;
		}
		if (diff < min_diff) {
			min_diff = diff;
			idx_tr = i;
		}
		if (diff > max_diff) {
			max_diff = diff;
			idx_bl = i;
		}
	}

	return {
		corners[idx_tl],
		corners[idx_tr],
		corners[idx_br],
		corners[idx_bl]
	};
}

} // namespace

namespace omr {

FrameStats analyzeRgba8888Frame(
		const std::uint8_t* pixels,
		int width,
		int height,
		int stride,
		int sample_step) {
	FrameStats stats{width, height, stride, 0.0, 0.0};

	if (pixels == nullptr || width <= 0 || height <= 0 || stride < width * 4) {
		return stats;
	}

	const int step = std::max(1, sample_step);
	double luma_sum = 0.0;
	int pixel_count = 0;
	int dark_count = 0;

	for (int y = 0; y < height; y += step) {
		const std::uint8_t* row = pixels + (y * stride);
		for (int x = 0; x < width; x += step) {
			const int idx = x * 4;

			const std::uint8_t r = row[idx + 0];
			const std::uint8_t g = row[idx + 1];
			const std::uint8_t b = row[idx + 2];

			const double luma = (0.299 * r) + (0.587 * g) + (0.114 * b);
			luma_sum += luma;
			pixel_count++;

			if (luma < 80.0) {
				dark_count++;
			}
		}
	}

	if (pixel_count > 0) {
		stats.average_luma = luma_sum / static_cast<double>(pixel_count);
		stats.dark_pixel_ratio = static_cast<double>(dark_count) / static_cast<double>(pixel_count);
	}

	return stats;
}

std::string toSummaryString(const FrameStats& stats) {
	std::ostringstream oss;
	oss << std::fixed << std::setprecision(4);
	oss << "JNI_OK"
		<< "|width=" << stats.width
		<< "|height=" << stats.height
		<< "|stride=" << stats.stride
		<< "|avg_luma=" << stats.average_luma
		<< "|dark_ratio=" << stats.dark_pixel_ratio;
	return oss.str();
}

ScanWarpResult scanAndWarpOmrPage(
		const cv::Mat& bgr,
		const cv::Size& output_size,
		bool collect_debug,
		const AnchorDetectionConfig& config) {
	ScanWarpResult result;

	if (bgr.empty()) {
		result.error = "input image is empty";
		return result;
	}

	cv::Mat working_bgr;
	if (bgr.channels() == 3) {
		working_bgr = bgr;
	} else if (bgr.channels() == 4) {
		cv::cvtColor(bgr, working_bgr, cv::COLOR_BGRA2BGR);
	} else if (bgr.channels() == 1) {
		cv::cvtColor(bgr, working_bgr, cv::COLOR_GRAY2BGR);
	} else {
		result.error = "unsupported channel count";
		return result;
	}

	if (output_size.width <= 0 || output_size.height <= 0) {
		result.error = "invalid output size";
		return result;
	}

	cv::Mat thresh = preprocessForAnchors(working_bgr);

	std::vector<std::vector<cv::Point>> all_contours;
	cv::findContours(thresh, all_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	if (collect_debug) {
		result.debug_image = working_bgr.clone();
		cv::drawContours(result.debug_image, all_contours, -1, cv::Scalar(255, 100, 0), 1);
	}

	std::vector<std::vector<cv::Point>> square_contours;
	square_contours.reserve(all_contours.size());
	for (const auto& contour : all_contours) {
		if (isSquareLikeContour(contour, working_bgr.size(), config)) {
			square_contours.push_back(contour);
		}
	}

	std::vector<cv::Point2f> centers = centersFromContours(square_contours);

	if (centers.size() == 3) {
		cv::Point2f guessed;
		if (inferFourthCorner(centers, &guessed)) {
			std::vector<cv::Point> rescued_contour;
			cv::Point2f rescued_center;
			if (rescueFourthContour(all_contours, guessed, config.rescue_radius_px, &rescued_contour, &rescued_center)) {
				square_contours.push_back(rescued_contour);
				centers.push_back(rescued_center);
				if (!result.debug_image.empty()) {
					cv::putText(
							result.debug_image,
							"RESCUED",
							guessed,
							cv::FONT_HERSHEY_PLAIN,
							1.2,
							cv::Scalar(0, 165, 255),
							2
					);
				}
			}
		}
	}

	if (!result.debug_image.empty()) {
		cv::drawContours(result.debug_image, square_contours, -1, cv::Scalar(0, 255, 255), 2);
	}

	if (centers.size() < 3) {
		result.error = "failed to detect at least 3 anchors";
		return result;
	}

	if (centers.size() == 3) {
		cv::Point2f inferred;
		if (!inferFourthCorner(centers, &inferred)) {
			result.error = "failed to infer fourth anchor";
			return result;
		}
		centers.push_back(inferred);
	}

	centers = selectFourExtremes(centers);
	centers = sortCornersTlTrBrBl(centers);

	if (centers.size() != 4) {
		result.error = "anchor corner selection failed";
		return result;
	}

	if (!result.debug_image.empty()) {
		for (int i = 0; i < 4; ++i) {
			cv::circle(result.debug_image, centers[i], 12, cv::Scalar(0, 255, 0), 2);
			cv::putText(
					result.debug_image,
					"A" + std::to_string(i),
					centers[i] + cv::Point2f(10.0f, -10.0f),
					cv::FONT_HERSHEY_SIMPLEX,
					0.7,
					cv::Scalar(0, 255, 0),
					2
			);
		}
	}

	const std::vector<cv::Point2f> destination = {
		cv::Point2f(0.0f, 0.0f),
		cv::Point2f(static_cast<float>(output_size.width), 0.0f),
		cv::Point2f(static_cast<float>(output_size.width), static_cast<float>(output_size.height)),
		cv::Point2f(0.0f, static_cast<float>(output_size.height))
	};

	const cv::Mat transform = cv::getPerspectiveTransform(centers, destination);
	cv::warpPerspective(working_bgr, result.warped, transform, output_size);

	if (result.warped.empty()) {
		result.error = "warpPerspective returned an empty image";
		return result;
	}

	result.success = true;
	result.corners = centers;
	result.error.clear();
	return result;
}

} // namespace omr
