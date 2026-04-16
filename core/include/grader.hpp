#pragma once
#include <opencv2/opencv.hpp>
#include <map>
#include <vector>
#include <string>

struct GradingResult {
    std::string exam_code;
    std::string student_id;
    std::map<int, std::string> questions;
};

class AnswerParser {
public:
    AnswerParser() = default;
    GradingResult parse_answers(const cv::Mat& warped_image, const std::string& debug_output_path = "");

private:
    cv::Mat binarize_crop(const cv::Mat& gray_crop);
    cv::Rect extract_info_box(const cv::Mat& gray, float expected_ratio);
    std::string parse_info_grid(const cv::Mat& gray, const cv::Rect& box, int cols, int rows, float fill_threshold, cv::Mat* debug_img = nullptr);
    std::vector<cv::Rect> extract_column_boxes(const cv::Mat& gray);
};
