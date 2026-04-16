#include "../include/utils.hpp"
#include <iomanip>

std::map<int, AnswerKey> load_answer_key(const std::string& csv_path) {
    std::map<int, AnswerKey> key_map;
    std::ifstream file(csv_path);
    std::string line, word;
    
    if(!file.is_open()) {
        std::cerr << "Failed to open answer key: " << csv_path << std::endl;
        return key_map;
    }
    
    getline(file, line); // header
    
    while(getline(file, line)) {
        std::stringstream ss(line);
        std::vector<std::string> row;
        while(getline(ss, word, ',')) { row.push_back(word); }
        if(row.size() >= 2) {
            try {
                AnswerKey ak;
                ak.order = std::stoi(row[0]);
                ak.key = row[1];
                ak.score = (row.size() >= 3 && !row[2].empty()) ? std::stof(row[2]) : 1.0f;
                key_map[ak.order] = ak;
            } catch(...) {}
        }
    }
    return key_map;
}

void export_to_csv(const std::string& output_path, const std::string& exam_code, const std::string& student_id, const std::map<int, ParsedAnswer>& results) {
    std::ofstream file(output_path);
    if(!file.is_open()) {
        std::cerr << "Failed to open output CSV: " << output_path << std::endl;
        return;
    }
    file << "Exam Code," << exam_code << ",\n";
    file << "Student ID," << student_id << ",\n\n";
    file << "Question,Parsed,Correct\n";
    
    for(const auto& pair : results) {
        std::string status = pair.second.is_correct ? "Yes" : "No";
        file << pair.first << "," << pair.second.answer << "," << status << "\n";
    }
    std::cout << "Exported results to " << output_path << std::endl;
}
