import cv2
import numpy as np
import json
import os
import csv
from typing import Dict, List, Tuple, Any, Optional

class AnswerParser:
    OPTION_LABELS = ['A', 'B', 'C', 'D']

    def __init__(self, template_json_path: str):
        self.template_json_path = template_json_path
        with open(template_json_path, 'r') as f:
            self.template = json.load(f)

    def _binarize_crop(self, gray_crop: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(gray_crop, (5, 5), 0)
        _, binary_inv = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary_inv
        
    def _extract_info_box(self, gray: np.ndarray, expected_ratio: float) -> Optional[Tuple[int, int, int, int]]:
        h_img, w_img = gray.shape[:2]
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_box = None
        min_diff = float('inf')
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if area > h_img * w_img * 0.001 and y < h_img * 0.4:
                ratio = h / max(1, float(w))
                diff = abs(ratio - expected_ratio)
                if diff < 0.5 and diff < min_diff:
                    min_diff = diff
                    best_box = (x, y, w, h)
        return best_box

    def _parse_info_grid(self, gray: np.ndarray, bx: int, by: int, bw: int, bh: int, cols: int, rows: int, debug_img: Optional[np.ndarray], fill_threshold: float) -> str:
        box_w_ref = cols * 6.0
        box_h_ref = rows * 5.0 + 19.0
        
        val = ""
        for i in range(cols):
            cx_pct = (3.0 + i * 6.0) / box_w_ref
            cx = bx + int(cx_pct * bw)
            
            option_fill_ratios = []
            option_detected = []
            
            for j in range(rows):
                cy_pct = (19.0 + j * 5.0) / box_h_ref
                cy = by + int(cy_pct * bh)
                
                cell_w = int((5.0 / box_w_ref) * bw)
                cell_h = int((4.0 / box_h_ref) * bh)
                
                x1 = max(0, cx - cell_w//2)
                y1 = max(0, cy - cell_h//2)
                x2 = min(gray.shape[1], cx + cell_w//2)
                y2 = min(gray.shape[0], cy + cell_h//2)
                
                crop_gray = gray[y1:y2, x1:x2]
                fill_ratio = 0.0
                r_inner = max(2, int(min(cell_w, cell_h) * 0.35))
                
                if crop_gray.size > 0:
                    crop_bin = self._binarize_crop(crop_gray)
                    local_cx, local_cy = cx - x1, cy - y1
                    mask = np.zeros(crop_bin.shape, dtype=np.uint8)
                    cv2.circle(mask, (local_cx, local_cy), r_inner, 255, -1)
                    fill_pixels = cv2.countNonZero(cv2.bitwise_and(crop_bin, crop_bin, mask=mask))
                    total_pixels = max(1, cv2.countNonZero(mask))
                    fill_ratio = fill_pixels / total_pixels
                    
                option_fill_ratios.append(fill_ratio)
                option_detected.append((cx, cy, x1, y1, x2, y2, r_inner))
                
                if debug_img is not None:
                    cv2.rectangle(debug_img, (x1, y1), (x2, y2), (255, 0, 0), 1)
                    cv2.circle(debug_img, (cx, cy), r_inner, (0, 255, 0), 1)
            
            if max(option_fill_ratios) < fill_threshold:
                val += "?"
            else:
                best_idx = int(np.argmax(option_fill_ratios))
                val += str(best_idx)
                if debug_img is not None:
                    cx, cy, _, _, _, _, r_inner = option_detected[best_idx]
                    cv2.circle(debug_img, (cx, cy), int(r_inner * 1.5), (0, 0, 255), 2)
                    
        return val

    def extract_column_boxes(self, gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h_img, w_img = binary.shape[:2]
        
        column_boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if area > h_img * w_img * 0.01:
                ratio = h / max(1, w)
                if 2.4 <= ratio <= 2.8 and y > h_img * 0.4:
                    column_boxes.append((x, y, w, h))
                    
        column_boxes = sorted(column_boxes, key=lambda b: b[0])
        return column_boxes

    def parse_answers(self, warped_image: np.ndarray, page_num: int = 1, debug_output_path: str = None) -> Dict[str, Any]:
        h, w = warped_image.shape[:2]
        gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
        
        debug_img = None
        if debug_output_path:
            debug_img = warped_image.copy()

        answers = {
            "exam_code": "",
            "student_id": "",
            "questions": {}
        }
        
        # Parse Info boxes layout first
        exam_code_box = self._extract_info_box(gray, expected_ratio=3.83) # 69 / 18
        if exam_code_box:
            if debug_img is not None:
                cvbx, cxby, cvbw, cvbh = exam_code_box
                cv2.rectangle(debug_img, (cvbx, cxby), (cvbx+cvbw, cxby+cvbh), (0, 255, 0), 2)
            answers["exam_code"] = self._parse_info_grid(gray, *exam_code_box, cols=3, rows=10, debug_img=debug_img, fill_threshold=0.35)
            
        student_id_box = self._extract_info_box(gray, expected_ratio=1.15) # 69 / 60
        if student_id_box:
            if debug_img is not None:
                cvbx, cxby, cvbw, cvbh = student_id_box
                cv2.rectangle(debug_img, (cvbx, cxby), (cvbx+cvbw, cxby+cvbh), (0, 255, 0), 2)
            answers["student_id"] = self._parse_info_grid(gray, *student_id_box, cols=10, rows=10, debug_img=debug_img, fill_threshold=0.35)


        column_boxes = self.extract_column_boxes(gray)
        if not column_boxes:
            print("No column boxes found!")
            if debug_img is not None:
                cv2.imwrite(debug_output_path, debug_img)
            return answers

        q_idx = 1
        
        fill_threshold = 0.45
        multiple_second_min_ratio = 0.40
        multiple_gap_max_ratio = 0.15

        for col_i, (bx, by, bw, bh) in enumerate(column_boxes):
            if debug_img is not None:
                cv2.rectangle(debug_img, (bx, by), (bx+bw, by+bh), (0, 255, 255), 2)
            
            rows_per_col = 20
            
            for r in range(rows_per_col):
                cy_pct = (4.0 + r * 5.5) / 115.0
                cy = by + int(cy_pct * bh)
                
                option_fill_ratios = []
                
                for j in range(4):
                    cx_pct = (10.0 + j * 8.0) / 42.0
                    cx = bx + int(cx_pct * bw)
                    
                    cell_w = int((7.0 / 42.0) * bw)
                    cell_h = int((4.5 / 115.0) * bh)
                    
                    x1 = max(0, cx - cell_w//2)
                    y1 = max(0, cy - cell_h//2)
                    x2 = min(w, cx + cell_w//2)
                    y2 = min(h, cy + cell_h//2)
                    
                    crop_gray = gray[y1:y2, x1:x2]
                    if crop_gray.size > 0:
                        crop_bin = self._binarize_crop(crop_gray)
                        local_cx, local_cy = cx - x1, cy - y1
                        r_inner = max(2, int(min(cell_w, cell_h) * 0.35))
                        mask = np.zeros(crop_bin.shape, dtype=np.uint8)
                        cv2.circle(mask, (local_cx, local_cy), r_inner, 255, -1)
                        
                        fill_pixels = cv2.countNonZero(cv2.bitwise_and(crop_bin, crop_bin, mask=mask))
                        total_pixels = cv2.countNonZero(mask)
                        fill_ratio = fill_pixels / max(1, total_pixels)
                    else:
                        fill_ratio = 0.0
                        
                    option_fill_ratios.append(fill_ratio)
                    
                    if debug_img is not None:
                        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (255, 0, 0), 1)
                        cv2.circle(debug_img, (cx, cy), r_inner, (0, 255, 0), 1)
                        cv2.putText(debug_img, f"{fill_ratio:.2f}", (x1, y1-2), cv2.FONT_HERSHEY_PLAIN, 0.6, (0, 255, 0), 1)

                if max(option_fill_ratios) < 0.1 and sum(option_fill_ratios) < 0.2:
                    continue

                ranked = sorted(enumerate(option_fill_ratios), key=lambda p: p[1], reverse=True)
                top_idx, top_ratio = ranked[0]
                second_idx, second_ratio = ranked[1]
                
                selected_indices = []
                if top_ratio < fill_threshold:
                    answers["questions"][q_idx] = ""
                elif second_ratio >= multiple_second_min_ratio and (top_ratio - second_ratio) <= multiple_gap_max_ratio:
                    answers["questions"][q_idx] = "M"
                    selected_indices = [top_idx, second_idx]
                else:
                    answers["questions"][q_idx] = self.OPTION_LABELS[top_idx]
                    selected_indices = [top_idx]
                    
                if debug_img is not None:
                    for idx in selected_indices:
                        cx_pct = (10.0 + idx * 8.0) / 42.0
                        cx = bx + int(cx_pct * bw)
                        r_out = int(min(cell_w, cell_h) * 0.45)
                        cv2.circle(debug_img, (cx, cy), r_out, (0, 0, 255), 2)
                        
                q_idx += 1
                if q_idx > 45: # max for this template
                    break
        
        if debug_img is not None:
            cv2.imwrite(debug_output_path, debug_img)
            
        return answers

def export_to_csv(parsed_data: Dict[str, Any], output_csv: str):
    answers = parsed_data.get('questions', parsed_data)
    exam_code = parsed_data.get('exam_code', '')
    student_id = parsed_data.get('student_id', '')
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        if exam_code:
            writer.writerow(["Exam Code", exam_code, ""])
        if student_id:
            writer.writerow(["Student ID", student_id, ""])
        writer.writerow([])
        writer.writerow(["Question", "Answer", "Status"])
        for q_num in sorted(answers.keys()):
            ans = answers[q_num]
            if ans == "": status = "Not Chosen"
            elif ans == "M" or len(ans) > 1: status = "Multiple Chosen"
            else: status = "Chosen"
            writer.writerow([q_num, ans, status])
    print(f"Exported answers to {output_csv}")


if __name__ == "__main__":
    template_path = "test/templates/Professional_OMR_45.json"
    warped_img_path = "outputs/scanner_debug_ver2/normal_warped.png"
    output_csv = "outputs/scanner_debug_ver2/normal_warped.csv"
    
    if os.path.exists(warped_img_path):
        parser = AnswerParser(template_path)
        img = cv2.imread(warped_img_path)
        
        print("Parsing answers using grid strategy...")
        debug_path = output_csv.replace(".csv", "_debug.png")
        answers = parser.parse_answers(img, page_num=1, debug_output_path=debug_path)
        
        print(f"Exam Code:  {answers.get('exam_code', '')}")
        print(f"Student ID: {answers.get('student_id', '')}")
        for q, a in answers.get("questions", {}).items():
            print(f"Q{q}: {a if a else 'No Answer'}")
            
        export_to_csv(answers, output_csv)
        print(f"Debug image saved to {debug_path}")
    else:
        print(f"Image {warped_img_path} not found.")

