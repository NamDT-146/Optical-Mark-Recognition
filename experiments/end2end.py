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

    corners = None
    processed = scanner.preprocess_image(img)
    square_contours = scanner.find_square_contours(processed)
    
    centers = []
    for cnt in square_contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centers.append((cx, cy))
            
    if len(centers) == 4:
        corners = scanner.sort_corners(centers)
    elif len(centers) == 3:
        corners = scanner.infer_fourth_anchor(centers)
        corners = scanner.sort_corners(corners)
        
    if not corners or len(corners) < 4:
        print("Failed to detect anchoring corners.")
        # Fall back to using the image as-is (e.g. if it's already warped)
        warped_img = img
    else:
        # Define output size based on template
        with open(template_path, 'r') as f:
            template = json.load(f)
        page_size = template.get("page_size_px", {"width": 2100, "height": 2970})
        expected_w = int(page_size["width"])
        expected_h = int(page_size["height"])
        
        dst_corners = np.float32([
            [0, 0],
            [expected_w, 0],
            [expected_w, expected_h],
            [0, expected_h]
        ])
        src_corners = np.float32(corners)
        M_transform = cv2.getPerspectiveTransform(src_corners, dst_corners)
        warped_img = cv2.warpPerspective(img, M_transform, (expected_w, expected_h))

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
    
    # Using the pre-warped image for now to test the parsing and scoring directly,
    # as the scanner might need tuning for this specific image.
    img_path = "outputs/scanner_debug_ver2/normal_warped.png" 
    out_csv = "outputs/scanner_debug_ver2/final_results.csv"
    
    parse_and_score(img_path, template_path, key_path, out_csv)
