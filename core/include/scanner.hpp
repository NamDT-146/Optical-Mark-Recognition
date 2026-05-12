#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace omr {

struct FrameStats {
	int width;
	int height;
	int stride;
	double average_luma;
	double dark_pixel_ratio;
};

struct AnchorDetectionConfig {
	double min_area = 100.0;
	double max_area = 8000.0;
	double aspect_ratio_tolerance = 0.3;
	double min_solidity = 0.8;
	double min_extent = 0.7;
	int rescue_radius_px = 200;
};

struct ScanWarpResult {
	bool success = false;
	std::string error;
	std::vector<cv::Point2f> corners;
	cv::Mat warped;
	cv::Mat debug_image;
};

FrameStats analyzeRgba8888Frame(
		const std::uint8_t* pixels,
		int width,
		int height,
		int stride,
		int sample_step = 1
);

std::string toSummaryString(const FrameStats& stats);

ScanWarpResult scanAndWarpOmrPage(
		const cv::Mat& bgr,
		const cv::Size& output_size = cv::Size(2100, 2970),
		bool collect_debug = false,
		const AnchorDetectionConfig& config = AnchorDetectionConfig()
);

} // namespace omr
