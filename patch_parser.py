with open("experiments/answer_parser/answer_parser.py", "r") as f:
    lines = f.readlines()

new_lines = []
in_class = False
for line in lines:
    if line.strip() == "def parse_answers(self, warped_image: np.ndarray, page_num: int = 1, debug_output_path: str = None) -> Dict[int, str]:":
        new_lines.append("""    def _extract_info_box(self, gray: np.ndarray, expected_ratio: float) -> Optional[Tuple[int, int, int, int]]:
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
                best_idx = np.argmax(option_fill_ratios)
                val += str(best_idx)
                if debug_img is not None:
                    cx, cy, _, _, _, _, r_inner = option_detected[best_idx]
                    cv2.circle(debug_img, (cx, cy), int(r_inner * 1.5), (0, 0, 255), 2)
                    
        return val

    def parse_answers(self, warped_image: np.ndarray, page_num: int = 1, debug_output_path: str = None) -> Dict[str, Any]:
""")
    elif line.strip() == "answers = {}":
        new_lines.append("""        
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
""")
    elif line.strip() == "answers[q_num] = \"\"":
        new_lines.append("                    answers[\"questions\"][q_num] = \"\"\n")
    elif line.strip() == "answers[q_num] = \"M\"":
        new_lines.append("                    answers[\"questions\"][q_num] = \"M\"\n")
    elif line.strip() == "answers[q_num] = options[top_idx]":
        new_lines.append("                    answers[\"questions\"][q_num] = options[top_idx]\n")
    elif "return answers" in line:
        new_lines.append(line)
    elif "def export_to_csv(" in line:
        new_lines.append("def export_to_csv(parsed_data: Dict[str, Any], output_csv: str):\n")
        new_lines.append("    answers = parsed_data.get('questions', parsed_data)\n")
        new_lines.append("    exam_code = parsed_data.get('exam_code', '')\n")
        new_lines.append("    student_id = parsed_data.get('student_id', '')\n")
    elif "writer.writerow([\"Question\", \"Answer\", \"Status\"])" in line:
        new_lines.append("        if exam_code:\n")
        new_lines.append("            writer.writerow([\"Exam Code\", exam_code, \"\"])\n")
        new_lines.append("        if student_id:\n")
        new_lines.append("            writer.writerow([\"Student ID\", student_id, \"\"])\n")
        new_lines.append("        writer.writerow([])\n")
        new_lines.append(line)
    elif "for q, a in answers.items():" in line:
        new_lines.append("""            print(f"Exam Code:  {answers.get('exam_code', '')}")
            print(f"Student ID: {answers.get('student_id', '')}")
            for q, a in answers.get("questions", {}).items():\n""")
    else:
        new_lines.append(line)

with open("experiments/answer_parser/answer_parser.py", "w") as f:
    f.writelines(new_lines)
