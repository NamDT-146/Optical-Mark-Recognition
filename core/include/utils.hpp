#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <map>
#include <iostream>
#include <fstream>
#include <sstream>

struct AnswerKey {
    int order;
    std::string key;
    float score;
};

struct ParsedAnswer {
    std::string answer;
    bool is_correct;
};

std::map<int, AnswerKey> load_answer_key(const std::string& csv_path);
void export_to_csv(const std::string& output_path, const std::string& exam_code, const std::string& student_id, const std::map<int, ParsedAnswer>& results);
