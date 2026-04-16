#include <iostream>
#include <string>
#include <map>
#include "../include/utils.hpp"
#include "../include/grader.hpp"

int main() {
    std::string img_path = "../../outputs/scanner_debug_ver2/normal_warped.png";
    std::string key_path = "../../test/keys/TEST_1.csv";
    std::string out_csv = "../../outputs/scanner_debug_ver2/final_results_cpp.csv";
    std::string debug_path = "../../outputs/scanner_debug_ver2/cpp_debug.png";

    cv::Mat warped_img = cv::imread(img_path);
    if(warped_img.empty()) {
        std::cerr << "Failed to read image: " << img_path << std::endl;
        return -1;
    }

    std::cout << "Processing image: " << img_path << std::endl;

    AnswerParser parser;
    GradingResult parsed = parser.parse_answers(warped_img, debug_path);

    std::map<int, AnswerKey> answer_key = load_answer_key(key_path);
    int total_score = 0;
    int max_possible_score = answer_key.size();
    int correct_count = 0;

    std::map<int, ParsedAnswer> results;

    for (const auto& pair : answer_key) {
        int q_num = pair.first;
        std::string correct_ans = pair.second.key;
        std::string parsed_ans = parsed.questions[q_num];

        bool is_correct = false;
        if (parsed_ans == "M") {
            // Multiple is FALSE
        } else if (parsed_ans == correct_ans) {
            is_correct = true;
            correct_count++;
        }

        results[q_num] = {parsed_ans, is_correct};
    }

    float final_score_10 = (max_possible_score > 0) ? ((float)correct_count / max_possible_score) * 10.0f : 0.0f;

    std::cout << "\n--- SCORES (C++) ---\n";
    std::cout << "Correct Answers: " << correct_count << "/" << max_possible_score << std::endl;
    std::cout << "Scaled Score (out of 10): " << final_score_10 << std::endl;

    export_to_csv(out_csv, parsed.exam_code, parsed.student_id, results);

    return 0;
}
