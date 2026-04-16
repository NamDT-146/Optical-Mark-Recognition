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

        page_size = self.template.get("page_size_px", {})
        self.reference_width = int(page_size.get("width", 2100))
        self.reference_height = int(page_size.get("height", 2970))

    def _get_questions_roi(self, page_num: int) -> Optional[Dict[str, Any]]:
        for page in self.template.get("pages", []):
            if page.get("page_number") == page_num:
                return page.get("rois", {}).get("questions")
        return None

    def _coordinates_from_bubble_centers(self, bubble_centers: Dict[str, Any]) -> Dict[int, List[Tuple[int, int]]]:
        """
        Parse exact exported circle centers from JSON.
        Supports:
        - {"1": {"A": {"x": ..., "y": ...}, ...}}
        - {"1": [[x, y], [x, y], [x, y], [x, y]]}
        """
        questions_coords: Dict[int, List[Tuple[int, int]]] = {}

        for q_key, option_data in bubble_centers.items():
            try:
                q_num = int(q_key)
            except (TypeError, ValueError):
                continue

            bubble_coords: List[Tuple[int, int]] = []

            if isinstance(option_data, dict):
                for label in self.OPTION_LABELS:
                    point = option_data.get(label)
                    if not isinstance(point, dict) or "x" not in point or "y" not in point:
                        bubble_coords = []
                        break
                    bubble_coords.append((int(round(point["x"])), int(round(point["y"]))))

            elif isinstance(option_data, list):
                for point in option_data[:4]:
                    if not isinstance(point, (list, tuple)) or len(point) < 2:
                        bubble_coords = []
                        break
                    bubble_coords.append((int(round(point[0])), int(round(point[1]))))

            if len(bubble_coords) == 4:
                questions_coords[q_num] = bubble_coords

        return dict(sorted(questions_coords.items()))

    def _coordinates_from_geometry(self, q_dict: Dict[str, Any], page_num: int) -> Dict[int, List[Tuple[int, int]]]:
        """
        Calculate circle centers from exported geometry.
        Falls back to legacy defaults for old templates.
        """
        questions_coords: Dict[int, List[Tuple[int, int]]] = {}

        start_q = int(q_dict["start_q"])
        num_qs = int(q_dict["num_qs"])
        base_x = float(q_dict["x"])
        base_y = float(q_dict["y"])

        row_h = float(q_dict.get("row_h", 55))
        rows_per_col = int(q_dict.get("rows_per_col", 20 if page_num == 1 else 35))
        col_step = float(q_dict.get("col_step", 430))
        option_start_x = float(q_dict.get("option_start_x", 100))
        option_step_x = float(q_dict.get("option_step_x", 80))
        bubble_center_y_offset = float(q_dict.get("bubble_center_y_offset", -10))

        for q_idx in range(num_qs):
            q_num = start_q + q_idx

            c = q_idx // rows_per_col
            r = q_idx % rows_per_col

            curr_x = base_x + (c * col_step)
            cy = base_y + (r * row_h) + bubble_center_y_offset

            bubble_coords: List[Tuple[int, int]] = []
            for j in range(4):  # A, B, C, D
                bx = curr_x + option_start_x + (j * option_step_x)
                bubble_coords.append((int(round(bx)), int(round(cy))))

            questions_coords[q_num] = bubble_coords

        return questions_coords

    def get_question_coordinates(self, q_dict: Dict[str, Any], page_num: int) -> Dict[int, List[Tuple[int, int]]]:
        """
        Calculate pixel coordinates for all question bubbles based on the given template ROI.
        Returns a dictionary mapping question number to a list of (x, y) coordinates for options A, B, C, D.
        """
        bubble_centers = q_dict.get("bubble_centers")
        if isinstance(bubble_centers, dict) and bubble_centers:
            exact_coords = self._coordinates_from_bubble_centers(bubble_centers)
            if exact_coords:
                return exact_coords

        return self._coordinates_from_geometry(q_dict, page_num)

    def _binarize_crop(self, gray_crop: np.ndarray) -> np.ndarray:
        """
        Binarize a local crop. Local Otsu thresholding is robust when lighting differs across page.
        """
        blurred = cv2.GaussianBlur(gray_crop, (5, 5), 0)
        _, binary_inv = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary_inv

    def _detect_circle_near_expected(
        self,
        gray_image: np.ndarray,
        expected_x: int,
        expected_y: int,
        expected_radius: int,
    ) -> Tuple[int, int, int]:
        """
        Detect a bubble circle near expected metadata coordinates on a local binarized crop.
        Falls back to expected values if detection fails.
        """
        h, w = gray_image.shape[:2]
        search_pad = max(6, int(round(expected_radius * 0.6)))
        search_radius = expected_radius + search_pad

        x1 = max(0, expected_x - search_radius)
        y1 = max(0, expected_y - search_radius)
        x2 = min(w, expected_x + search_radius)
        y2 = min(h, expected_y + search_radius)

        if x2 <= x1 or y2 <= y1:
            return expected_x, expected_y, expected_radius

        crop_gray = gray_image[y1:y2, x1:x2]
        if crop_gray.size == 0:
            return expected_x, expected_y, expected_radius

        crop_bin = self._binarize_crop(crop_gray)

        expected_local_x = expected_x - x1
        expected_local_y = expected_y - y1

        min_r = max(4, int(round(expected_radius * 0.55)))
        max_r = max(min_r + 2, int(round(expected_radius * 1.55)))

        best_circle: Optional[Tuple[int, int, int]] = None
        best_score = float("inf")

        contours, _ = cv2.findContours(crop_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area <= 0:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue

            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            if radius < min_r or radius > max_r:
                continue

            circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
            if circularity < 0.45:
                continue

            distance = float(np.hypot(cx - expected_local_x, cy - expected_local_y))
            if distance > (expected_radius + search_pad):
                continue

            score = distance + (1.5 * abs(radius - expected_radius)) + (max(0.0, 0.75 - circularity) * 25.0)
            if score < best_score:
                best_score = score
                best_circle = (
                    int(round(cx)) + x1,
                    int(round(cy)) + y1,
                    int(round(radius)),
                )

        if best_circle is not None:
            return best_circle

        # Fallback to Hough transform if contour filtering cannot isolate a clear circle.
        crop_blur = cv2.GaussianBlur(crop_gray, (5, 5), 0)
        circles = cv2.HoughCircles(
            crop_blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(8, expected_radius),
            param1=80,
            param2=12,
            minRadius=min_r,
            maxRadius=max_r,
        )

        if circles is not None and len(circles) > 0:
            best_hough: Optional[Tuple[int, int, int]] = None
            best_hough_score = float("inf")
            for cx, cy, radius in circles[0]:
                distance = float(np.hypot(cx - expected_local_x, cy - expected_local_y))
                score = distance + abs(radius - expected_radius)
                if score < best_hough_score:
                    best_hough_score = score
                    best_hough = (
                        int(round(cx)) + x1,
                        int(round(cy)) + y1,
                        int(round(radius)),
                    )

            if best_hough is not None:
                return best_hough

        return expected_x, expected_y, expected_radius

    def _compute_circle_statistics(
        self,
        gray_image: np.ndarray,
        center_x: int,
        center_y: int,
        radius: int,
    ) -> Tuple[float, float, Tuple[int, int, int, int]]:
        """
        Compute fill ratio and ring ratio for a detected bubble.
        - fill ratio: darkness inside inner area.
        - ring ratio: darkness on circle ring band.
        """
        h, w = gray_image.shape[:2]
        outer_r = max(6, int(round(radius * 1.25)))

        x1 = max(0, center_x - outer_r)
        y1 = max(0, center_y - outer_r)
        x2 = min(w, center_x + outer_r)
        y2 = min(h, center_y + outer_r)

        if x2 <= x1 or y2 <= y1:
            return 0.0, 0.0, (x1, y1, x2, y2)

        crop_gray = gray_image[y1:y2, x1:x2]
        if crop_gray.size == 0:
            return 0.0, 0.0, (x1, y1, x2, y2)

        crop_bin = self._binarize_crop(crop_gray)

        local_cx = center_x - x1
        local_cy = center_y - y1
        if local_cx < 0 or local_cy < 0 or local_cx >= crop_bin.shape[1] or local_cy >= crop_bin.shape[0]:
            return 0.0, 0.0, (x1, y1, x2, y2)

        max_fit_r = min(
            local_cx,
            local_cy,
            crop_bin.shape[1] - 1 - local_cx,
            crop_bin.shape[0] - 1 - local_cy,
        )
        usable_r = min(radius, max_fit_r)
        if usable_r < 4:
            return 0.0, 0.0, (x1, y1, x2, y2)

        inner_fill_r = max(2, int(round(usable_r * 0.72)))
        ring_inner_r = max(inner_fill_r + 1, int(round(usable_r * 0.80)))
        ring_outer_r = max(ring_inner_r + 1, int(round(usable_r * 1.00)))

        fill_mask = np.zeros(crop_bin.shape, dtype=np.uint8)
        cv2.circle(fill_mask, (int(local_cx), int(local_cy)), int(inner_fill_r), 255, -1)

        outer_mask = np.zeros(crop_bin.shape, dtype=np.uint8)
        inner_ring_mask = np.zeros(crop_bin.shape, dtype=np.uint8)
        cv2.circle(outer_mask, (int(local_cx), int(local_cy)), int(ring_outer_r), 255, -1)
        cv2.circle(inner_ring_mask, (int(local_cx), int(local_cy)), int(ring_inner_r), 255, -1)
        ring_mask = cv2.subtract(outer_mask, inner_ring_mask)

        fill_total = cv2.countNonZero(fill_mask)
        ring_total = cv2.countNonZero(ring_mask)
        if fill_total == 0 or ring_total == 0:
            return 0.0, 0.0, (x1, y1, x2, y2)

        fill_pixels = cv2.countNonZero(cv2.bitwise_and(crop_bin, crop_bin, mask=fill_mask))
        ring_pixels = cv2.countNonZero(cv2.bitwise_and(crop_bin, crop_bin, mask=ring_mask))

        fill_ratio = fill_pixels / fill_total
        ring_ratio = ring_pixels / ring_total
        return fill_ratio, ring_ratio, (x1, y1, x2, y2)

    def _estimate_alignment_from_pairs(
        self,
        source_points: List[Tuple[float, float]],
        target_points: List[Tuple[float, float]],
        expected_radius: int,
    ) -> Optional[np.ndarray]:
        """
        Estimate a loose affine transform from matched source/target points.
        """
        if len(source_points) < 6 or len(target_points) < 6:
            return None

        src_np = np.array(source_points, dtype=np.float32)
        dst_np = np.array(target_points, dtype=np.float32)
        transform, _ = cv2.estimateAffinePartial2D(
            src_np,
            dst_np,
            method=cv2.RANSAC,
            ransacReprojThreshold=max(4.0, expected_radius * 0.9),
            maxIters=3000,
            confidence=0.97,
        )
        return transform

    def _transform_point(self, transform: Optional[np.ndarray], x: int, y: int) -> Tuple[int, int]:
        if transform is None:
            return x, y

        tx = (transform[0, 0] * x) + (transform[0, 1] * y) + transform[0, 2]
        ty = (transform[1, 0] * x) + (transform[1, 1] * y) + transform[1, 2]
        return int(round(tx)), int(round(ty))

    def parse_answers(self, warped_image: np.ndarray, page_num: int = 1, debug_output_path: str = None) -> Dict[int, str]:
        """
        Parse answer bubbles from the warped image.
        Returns a dictionary of marked options per question (e.g., {1: 'A', 2: 'B', 3: 'M', 4: ''})
        """
        h, w = warped_image.shape[:2]
        gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
        
        debug_img = None
        if debug_output_path:
            debug_img = warped_image.copy()

        # Locate the correct page ROI from template
        page_roi = self._get_questions_roi(page_num)

        if not page_roi:
            print(f"No questions ROI found for page {page_num}")
            return {}

        coords = self.get_question_coordinates(page_roi, page_num)
        if not coords:
            print(f"No question coordinates resolved for page {page_num}")
            return {}

        scale_x = w / max(1, self.reference_width)
        scale_y = h / max(1, self.reference_height)

        base_radius = float(page_roi.get("bubble_radius", 21))
        expected_radius = max(6, int(round(base_radius * ((scale_x + scale_y) * 0.5))))
        fill_threshold = float(page_roi.get("fill_threshold", 0.4))
        blank_fill_max = float(page_roi.get("blank_fill_max", 0.32))
        blank_ring_min = float(page_roi.get("blank_ring_min", 0.20))
        mark_min_ratio = float(page_roi.get("mark_min_ratio", max(0.52, fill_threshold)))
        blank_gap_min_ratio = float(page_roi.get("blank_gap_min_ratio", 0.28))
        single_gap_min_ratio = float(page_roi.get("single_gap_min_ratio", 0.16))
        multiple_second_min_ratio = float(page_roi.get("multiple_second_min_ratio", 0.52))
        multiple_gap_max_ratio = float(page_roi.get("multiple_gap_max_ratio", 0.12))
        multiple_strong_second_min_ratio = float(page_roi.get("multiple_strong_second_min_ratio", 0.68))

        # Pass 1: detect circles near metadata positions and collect blank anchors.
        anchor_source: List[Tuple[float, float]] = []
        anchor_target: List[Tuple[float, float]] = []
        blank_fill_samples: List[float] = []

        for bubbles in coords.values():
            for raw_bx, raw_cy in bubbles:
                expected_x = int(round(raw_bx * scale_x))
                expected_y = int(round(raw_cy * scale_y))

                det_x, det_y, det_r = self._detect_circle_near_expected(
                    gray,
                    expected_x,
                    expected_y,
                    expected_radius,
                )
                fill_ratio, ring_ratio, _ = self._compute_circle_statistics(gray, det_x, det_y, det_r)

                if fill_ratio <= blank_fill_max and ring_ratio >= blank_ring_min:
                    anchor_source.append((float(expected_x), float(expected_y)))
                    anchor_target.append((float(det_x), float(det_y)))
                    blank_fill_samples.append(float(fill_ratio))

                    if debug_img is not None:
                        cv2.circle(debug_img, (det_x, det_y), det_r, (255, 220, 0), 1)

        # Derive stricter mark thresholds from observed blank-circle behavior.
        if len(blank_fill_samples) >= 8:
            blank_median = float(np.median(blank_fill_samples))
            blank_p95 = float(np.percentile(blank_fill_samples, 95))
            mark_min_ratio = max(mark_min_ratio, min(0.95, blank_p95 + 0.16))
            blank_gap_min_ratio = max(blank_gap_min_ratio, min(0.60, (blank_p95 - blank_median) + 0.18))
            multiple_second_min_ratio = max(multiple_second_min_ratio, mark_min_ratio - 0.02)

        transform = self._estimate_alignment_from_pairs(anchor_source, anchor_target, expected_radius)
        
        answers = {}
        option_labels = self.OPTION_LABELS
        
        # Process each question
        for q_num, bubbles in coords.items():
            option_fill_ratios: List[float] = []
            option_detected: List[Tuple[int, int, int, int, int, int, int, int, int]] = []

            for j, (raw_bx, raw_cy) in enumerate(bubbles):
                expected_x = int(round(raw_bx * scale_x))
                expected_y = int(round(raw_cy * scale_y))

                aligned_x, aligned_y = self._transform_point(transform, expected_x, expected_y)
                det_x, det_y, det_r = self._detect_circle_near_expected(
                    gray,
                    aligned_x,
                    aligned_y,
                    expected_radius,
                )

                fill_ratio, _, (x1, y1, x2, y2) = self._compute_circle_statistics(gray, det_x, det_y, det_r)
                option_fill_ratios.append(float(fill_ratio))
                option_detected.append((det_x, det_y, det_r, x1, y1, x2, y2, aligned_x, aligned_y))
                
                if debug_img is not None:
                    cv2.rectangle(debug_img, (x1, y1), (x2, y2), (255, 0, 0), 1)
                    cv2.circle(debug_img, (det_x, det_y), det_r, (0, 255, 255), 1)
                    cv2.circle(debug_img, (aligned_x, aligned_y), 2, (0, 255, 0), -1)
                    cv2.putText(debug_img, f"{fill_ratio:.2f}", (x1, y1 - 2), cv2.FONT_HERSHEY_PLAIN, 0.6, (0, 255, 0), 1)

            if len(option_fill_ratios) == 0:
                answers[q_num] = "" # None chosen
            else:
                ranked = sorted(enumerate(option_fill_ratios), key=lambda p: p[1], reverse=True)
                top_idx, top_ratio = ranked[0]
                second_idx, second_ratio = ranked[1] if len(ranked) > 1 else (-1, 0.0)

                sorted_ratios = sorted(option_fill_ratios)
                blank_reference = float(np.mean(sorted_ratios[:3])) if len(sorted_ratios) >= 3 else float(np.min(option_fill_ratios))

                selected_indices: List[int] = []

                if top_ratio < mark_min_ratio:
                    answers[q_num] = "" # None chosen
                elif second_ratio >= multiple_strong_second_min_ratio:
                    answers[q_num] = "M" # Multiple chosen
                    selected_indices = [top_idx, second_idx]
                elif second_ratio >= multiple_second_min_ratio and (top_ratio - second_ratio) <= multiple_gap_max_ratio:
                    answers[q_num] = "M" # Multiple chosen
                    selected_indices = [top_idx, second_idx]
                elif (top_ratio - blank_reference) >= blank_gap_min_ratio and (top_ratio - second_ratio) >= single_gap_min_ratio:
                    answers[q_num] = option_labels[top_idx]
                    selected_indices = [top_idx]
                else:
                    answers[q_num] = "" # None chosen

                if debug_img is not None:
                    for idx in selected_indices:
                        if idx < 0 or idx >= len(option_detected):
                            continue
                        det_x, det_y, det_r, _, _, _, _, _, _ = option_detected[idx]
                        cv2.circle(debug_img, (det_x, det_y), det_r, (0, 0, 255), 2)
                
        if debug_img is not None:
            cv2.imwrite(debug_output_path, debug_img)
                
        return answers

def export_to_csv(answers: Dict[int, str], output_csv: str):
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Question", "Answer", "Status"])
        for q_num in sorted(answers.keys()):
            ans = answers[q_num]
            if ans == "":
                status = "Not Chosen"
            elif ans == "M" or len(ans) > 1:
                status = "Multiple Chosen"
            else:
                status = "Chosen"
            writer.writerow([q_num, ans, status])
    print(f"Exported answers to {output_csv}")

if __name__ == "__main__":
    template_path = "test/templates/Professional_OMR_45.json"
    warped_img_path = "outputs/scanner_debug_ver2/normal_warped.png" # Example
    output_csv = "outputs/scanner_debug_ver2/normal_warped.csv"
    
    if os.path.exists(warped_img_path):
        parser = AnswerParser(template_path)
        img = cv2.imread(warped_img_path)
        
        if img is not None:
            print("Parsing answers...")
            debug_path = output_csv.replace(".csv", "_debug.png")
            answers = parser.parse_answers(img, page_num=1, debug_output_path=debug_path)
            
            for q, a in answers.items():
                print(f"Q{q}: {a if a else 'No Answer'}")
                
            export_to_csv(answers, output_csv)
            print(f"Debug image saved to {debug_path}")
        else:
            print("Failed to read image")
    else:
        print(f"Image {warped_img_path} not found.")