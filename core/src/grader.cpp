#include "../include/grader.hpp"
#include <algorithm>
#include <iostream>

cv::Mat AnswerParser::binarize_crop(const cv::Mat& gray_crop) {
    cv::Mat blurred, binary_inv;
    cv::GaussianBlur(gray_crop, blurred, cv::Size(5, 5), 0);
    cv::threshold(blurred, binary_inv, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
    return binary_inv;
}

cv::Rect AnswerParser::extract_info_box(const cv::Mat& gray, float expected_ratio) {
    int h_img = gray.rows;
    int w_img = gray.cols;
    cv::Mat blurred, binary;
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);
    cv::threshold(blurred, binary, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
    
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    cv::Rect best_box(0,0,0,0);
    float min_diff = 1e9;
    
    for(const auto& cnt : contours) {
        cv::Rect rect = cv::boundingRect(cnt);
        float area = rect.width * rect.height;
        if(area > h_img * w_img * 0.001f && rect.y < h_img * 0.4f) {
            float ratio = (float)rect.height / std::max(1, rect.width);
            float diff = std::abs(ratio - expected_ratio);
            if(diff < 0.5f && diff < min_diff) {
                min_diff = diff;
                best_box = rect;
            }
        }
    }
    return best_box;
}

std::string AnswerParser::parse_info_grid(const cv::Mat& gray, const cv::Rect& box, int cols, int rows, float fill_threshold, cv::Mat* debug_img) {
    float box_w_ref = cols * 6.0f;
    float box_h_ref = rows * 5.0f + 19.0f;
    std::string val = "";
    
    for(int i = 0; i < cols; ++i) {
        float cx_pct = (3.0f + i * 6.0f) / box_w_ref;
        int cx = box.x + (int)(cx_pct * box.width);
        
        std::vector<float> option_fill_ratios;
        std::vector<cv::Rect> option_detected;
        std::vector<int> r_inners;
        
        for(int j = 0; j < rows; ++j) {
            float cy_pct = (19.0f + j * 5.0f) / box_h_ref;
            int cy = box.y + (int)(cy_pct * box.height);
            
            int cell_w = (int)((5.0f / box_w_ref) * box.width);
            int cell_h = (int)((4.0f / box_h_ref) * box.height);
            
            int x1 = std::max(0, cx - cell_w/2);
            int y1 = std::max(0, cy - cell_h/2);
            int x2 = std::min(gray.cols, cx + cell_w/2);
            int y2 = std::min(gray.rows, cy + cell_h/2);
            
            cv::Rect crop_rect(x1, y1, x2 - x1, y2 - y1);
            float fill_ratio = 0.0f;
            int r_inner = std::max(2, (int)(std::min(cell_w, cell_h) * 0.35f));
            
            if(crop_rect.width > 0 && crop_rect.height > 0) {
                cv::Mat crop_gray = gray(crop_rect);
                cv::Mat crop_bin = binarize_crop(crop_gray);
                int local_cx = cx - x1, local_cy = cy - y1;
                
                cv::Mat mask = cv::Mat::zeros(crop_bin.size(), CV_8UC1);
                cv::circle(mask, cv::Point(local_cx, local_cy), r_inner, cv::Scalar(255), -1);
                
                cv::Mat masked;
                cv::bitwise_and(crop_bin, crop_bin, masked, mask);
                int fill_pixels = cv::countNonZero(masked);
                int total_pixels = std::max(1, cv::countNonZero(mask));
                fill_ratio = (float)fill_pixels / total_pixels;
            }
            
            option_fill_ratios.push_back(fill_ratio);
            option_detected.push_back(cv::Rect(cx, cy, x2-x1, y2-y1));
            r_inners.push_back(r_inner);
            
            if(debug_img) {
                cv::rectangle(*debug_img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0), 1);
                cv::circle(*debug_img, cv::Point(cx, cy), r_inner, cv::Scalar(0, 255, 0), 1);
            }
        }
        
        auto max_it = std::max_element(option_fill_ratios.begin(), option_fill_ratios.end());
        int best_idx = std::distance(option_fill_ratios.begin(), max_it);
        if(*max_it < fill_threshold) {
            val += "?";
        } else {
            val += std::to_string(best_idx);
            if(debug_img) {
                int r_out = (int)(r_inners[best_idx] * 1.5f);
                cv::circle(*debug_img, cv::Point(option_detected[best_idx].x, option_detected[best_idx].y), r_out, cv::Scalar(0, 0, 255), 2);
            }
        }
    }
    return val;
}

std::vector<cv::Rect> AnswerParser::extract_column_boxes(const cv::Mat& gray) {
    cv::Mat blurred, binary;
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);
    cv::threshold(blurred, binary, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
    
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    int h_img = binary.rows;
    int w_img = binary.cols;
    std::vector<cv::Rect> column_boxes;
    
    for(const auto& cnt : contours) {
        cv::Rect rect = cv::boundingRect(cnt);
        float area = rect.width * rect.height;
        if(area > h_img * w_img * 0.01f) {
            float ratio = (float)rect.height / std::max(1, rect.width);
            if(ratio >= 2.4f && ratio <= 2.8f && rect.y > h_img * 0.4f) {
                column_boxes.push_back(rect);
            }
        }
    }
    std::sort(column_boxes.begin(), column_boxes.end(), [](const cv::Rect& a, const cv::Rect& b) {
        return a.x < b.x;
    });
    return column_boxes;
}

GradingResult AnswerParser::parse_answers(const cv::Mat& warped_image, const std::string& debug_output_path) {
    GradingResult result;
    cv::Mat gray;
    cv::cvtColor(warped_image, gray, cv::COLOR_BGR2GRAY);
    
    cv::Mat debug_img;
    if(!debug_output_path.empty()) {
        debug_img = warped_image.clone();
    }
    
    cv::Rect exam_code_box = extract_info_box(gray, 3.83f);
    if(exam_code_box.area() > 0) {
        if(!debug_img.empty()) {
            cv::rectangle(debug_img, exam_code_box, cv::Scalar(0, 255, 0), 2);
        }
        result.exam_code = parse_info_grid(gray, exam_code_box, 3, 10, 0.35f, !debug_img.empty() ? &debug_img : nullptr);
    }
    
    cv::Rect student_id_box = extract_info_box(gray, 1.15f);
    if(student_id_box.area() > 0) {
        if(!debug_img.empty()) {
            cv::rectangle(debug_img, student_id_box, cv::Scalar(0, 255, 0), 2);
        }
        result.student_id = parse_info_grid(gray, student_id_box, 10, 10, 0.35f, !debug_img.empty() ? &debug_img : nullptr);
    }
    
    std::vector<cv::Rect> column_boxes = extract_column_boxes(gray);
    if(column_boxes.empty()) {
        std::cerr << "No column boxes found!\n";
        if(!debug_output_path.empty()) cv::imwrite(debug_output_path, debug_img);
        return result;
    }
    
    int q_idx = 1;
    float fill_threshold = 0.45f;
    float multiple_second_min_ratio = 0.40f;
    float multiple_gap_max_ratio = 0.15f;
    std::vector<std::string> option_labels = {"A", "B", "C", "D"};
    
    for(const auto& box : column_boxes) {
        if(!debug_img.empty()) {
            cv::rectangle(debug_img, box, cv::Scalar(0, 255, 255), 2);
        }
        
        int rows_per_col = 20;
        for(int r = 0; r < rows_per_col; ++r) {
            float cy_pct = (4.0f + r * 5.5f) / 115.0f;
            int cy = box.y + (int)(cy_pct * box.height);
            
            std::vector<std::pair<int, float>> option_fill;
            int cell_w = (int)((7.0f / 42.0f) * box.width);
            int cell_h = (int)((4.5f / 115.0f) * box.height);
            int r_inner = std::max(2, (int)(std::min(cell_w, cell_h) * 0.35f));
            
            for(int j = 0; j < 4; ++j) {
                float cx_pct = (10.0f + j * 8.0f) / 42.0f;
                int cx = box.x + (int)(cx_pct * box.width);
                
                int x1 = std::max(0, cx - cell_w/2);
                int y1 = std::max(0, cy - cell_h/2);
                int x2 = std::min(gray.cols, cx + cell_w/2);
                int y2 = std::min(gray.rows, cy + cell_h/2);
                
                cv::Rect crop_rect(x1, y1, x2 - x1, y2 - y1);
                float fill_ratio = 0.0f;
                
                if(crop_rect.width > 0 && crop_rect.height > 0) {
                    cv::Mat crop_gray = gray(crop_rect);
                    cv::Mat crop_bin = binarize_crop(crop_gray);
                    int local_cx = cx - x1, local_cy = cy - y1;
                    
                    cv::Mat mask = cv::Mat::zeros(crop_bin.size(), CV_8UC1);
                    cv::circle(mask, cv::Point(local_cx, local_cy), r_inner, cv::Scalar(255), -1);
                    
                    cv::Mat masked;
                    cv::bitwise_and(crop_bin, crop_bin, masked, mask);
                    int fill_pixels = cv::countNonZero(masked);
                    int total_pixels = cv::countNonZero(mask);
                    fill_ratio = (float)fill_pixels / std::max(1, total_pixels);
                }
                
                option_fill.push_back({j, fill_ratio});
                
                if(!debug_img.empty()) {
                    cv::rectangle(debug_img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0), 1);
                    cv::circle(debug_img, cv::Point(cx, cy), r_inner, cv::Scalar(0, 255, 0), 1);
                }
            }
            
            float sum_fill = 0.0f;
            float max_fill = 0.0f;
            for(const auto& opt : option_fill) {
                sum_fill += opt.second;
                max_fill = std::max(max_fill, opt.second);
            }
            
            if(max_fill < 0.1f && sum_fill < 0.2f) {
                continue;
            }
            
            std::sort(option_fill.begin(), option_fill.end(), [](const auto& a, const auto& b){
                return a.second > b.second;
            });
            
            float top_ratio = option_fill[0].second;
            int top_idx = option_fill[0].first;
            float second_ratio = option_fill[1].second;
            int second_idx = option_fill[1].first;
            
            std::vector<int> selected_indices;
            if(top_ratio < fill_threshold) {
                result.questions[q_idx] = "";
            } else if (second_ratio >= multiple_second_min_ratio && (top_ratio - second_ratio) <= multiple_gap_max_ratio) {
                result.questions[q_idx] = "M";
                selected_indices = {top_idx, second_idx};
            } else {
                result.questions[q_idx] = option_labels[top_idx];
                selected_indices = {top_idx};
            }
            
            if(!debug_img.empty()) {
                for(int idx : selected_indices) {
                    float cx_pct = (10.0f + idx * 8.0f) / 42.0f;
                    int cx = box.x + (int)(cx_pct * box.width);
                    int r_out = (int)(std::min(cell_w, cell_h) * 0.45f);
                    cv::circle(debug_img, cv::Point(cx, cy), r_out, cv::Scalar(0, 0, 255), 2);
                }
            }
            q_idx++;
            if(q_idx > 45) break; 
        }
    }
    
    if(!debug_output_path.empty() && !debug_img.empty()) {
        cv::imwrite(debug_output_path, debug_img);
    }
    return result;
}
