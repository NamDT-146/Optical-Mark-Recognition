import cv2
import numpy as np
import os
import json
import csv
import sys

from images_parser.robust_scanner import RobustOMRScanner
from answer_parser.answer_parser import AnswerParser

def load_answer_key(csv_path: str) -> dict:
    key = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                q_num = int(row['Order'])
                ans = row['Key'].strip()
                score = float(row.get('Score', 1.0))
                key[q_num] = {'answer': ans, 'score': score}
            except ValueError:
                continue
    return key

def parse_and_score(image_path: str, template_path: str, key_path: str, output_csv: str = None):
    # 1. Scanning and Warping
    print(f"Processing image: {image_path}")
    scanner = RobustOMRScanner()
    img = cv2.imread(image_path)
    if img is None:
        print("Failed to read image.")
        return

    with open(template_path, 'r', encoding='utf-8') as f:
        template = json.load(f)
    page_size = template.get("page_size_px", {"width": 2100, "height": 2970})
    expected_w = int(page_size["width"])
    expected_h = int(page_size["height"])

    corners, scanner_debug = scanner.detect_anchors(img)
    if not corners or len(corners) < 4:
        print("Failed to detect anchoring corners.")
        warped_img = img
    else:
        warped_img, _ = scanner.warp_perspective(
            img,
            corners,
            output_size=(expected_w, expected_h)
        )

    warped_output_path = "outputs/scanner_debug_ver2/normal_warped.png"
    cv2.imwrite(warped_output_path, warped_img)
    if scanner_debug is not None:
        cv2.imwrite("outputs/scanner_debug_ver2/python_scanner_debug.png", scanner_debug)

    # 2. Parsing Answers
    parser = AnswerParser(template_path)
    
    debug_path = None
    if output_csv:
        debug_path = output_csv.replace(".csv", "_debug.png")
        
    answers = parser.parse_answers(warped_img, page_num=1, debug_output_path=debug_path)
    questions_parsed = answers.get('questions', answers)
    
    # 3. Scoring
    answer_key = load_answer_key(key_path)
    total_score = 0.0
    max_possible_score = len(answer_key)
    correct_count = 0
    
    # Scale to 10
    results = []
    
    for q_num, key_info in sorted(answer_key.items()):
        correct_ans = key_info['answer']
        parsed_ans = questions_parsed.get(q_num, "")
        
        is_correct = False
        if parsed_ans == "M":
            # Multiple answers is FALSE
            pass
        elif parsed_ans == correct_ans:
            is_correct = True
            correct_count += 1
            
        results.append({
            'Question': q_num,
            'Parsed': parsed_ans,
            'Key': correct_ans,
            'Correct': is_correct
        })
        
    final_score_10 = (correct_count / max_possible_score) * 10.0 if max_possible_score > 0 else 0.0
    
    print(f"\n--- SCORES ---")
    print(f"Correct Answers: {correct_count}/{max_possible_score}")
    print(f"Scaled Score (out of 10): {final_score_10:.2f}")
    
    if output_csv:
        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Exam Code", answers.get('exam_code', ''), ""])
            writer.writerow(["Student ID", answers.get('student_id', ''), ""])
            writer.writerow([])
            writer.writerow(["Question", "Parsed", "Key", "Correct"])
            for r in results:
                writer.writerow([r['Question'], r['Parsed'], r['Key'], "Yes" if r['Correct'] else "No"])
        print(f"Exported results to {output_csv}")

if __name__ == "__main__":
    template_path = "test/templates/Professional_OMR_45.json"
    key_path = "test/keys/TEST_1.csv"

    img_path = "test/images/ver2/flash.jpg"
    out_csv = "outputs/scanner_debug_ver2/final_results.csv"
    
    parse_and_score(img_path, template_path, key_path, out_csv)
