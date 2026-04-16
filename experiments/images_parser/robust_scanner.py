import cv2
import numpy as np
from typing import Optional, Tuple, List

class RobustOMRScanner:
    """
    Robust OMR scanner using geometric shape detection instead of color filtering.
    Handles real-world conditions: B&W printing, shadows, poor lighting.
    """
    
    def __init__(self, anchor_size_range=(100, 2000), aspect_ratio_tolerance=0.2, min_solidity=0.9):
        """
        Args:
            anchor_size_range: (min_area, max_area) for anchor squares in pixels
            aspect_ratio_tolerance: tolerance for square aspect ratio (0.8-1.2 = 0.2)
            min_solidity: minimum ratio of contour area to convex hull area (default > 0.9 for solid squares)
        """
        self.anchor_size_range = anchor_size_range
        self.aspect_ratio_tolerance = aspect_ratio_tolerance
        self.min_solidity = min_solidity
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocessing: Grayscale -> CLAHE -> Gaussian Blur -> Adaptive Thresholding.
        
        Added CLAHE to boost local contrast before thresholding, which helps 
        in very dark/shaded areas where global contrast is low.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # This is CRITICAL when the background is dark (shade) or inconsistent
        clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Gaussian blur to reduce noise - REDUCED to 3x3 to avoid smearing tiny anchors in far images
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # --- FIX 1: DYNAMIC BLOCK SIZE ---
        # Ensure the block size is large enough to encapsulate the entire black square
        # even on very high-resolution images.
        h, w = gray.shape
        dynamic_block_size = int(min(h, w) * 0.05) # 5% of the shortest image side
        if dynamic_block_size % 2 == 0: 
            dynamic_block_size += 1 # Must be an odd number
        dynamic_block_size = max(31, dynamic_block_size) # Floor it at 31 (was 51)
        
        # Adaptive thresholding: 
        # REDUCED C from 5 to 3: Be more sensitive to "gray" smudges (small black boxes seen from far away)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, blockSize=dynamic_block_size, C=3
        )
        
        # REMOVED MORPH_OPEN: Erode -> Dilate completely wipes out anchors smaller than 3x3 pixels!
        # Only use MORPH_CLOSE (Dilate -> Erode) to fill holes in anchors
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        return thresh
    
    def find_square_contours(self, thresh: np.ndarray, debug_image: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """
        Find contours and filter strictly for SQUARE shapes.
        Logs why each contour was rejected for debugging.
        """
        h, w = thresh.shape[:2]
        img_area = h * w
        
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # # Change cv2.RETR_EXTERNAL to cv2.RETR_LIST
        # contours, _ = cv2.findContours(
        #         thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        # )
        
        # --- VISUALIZE EVERY RAW CONTOUR ---
        if debug_image is not None:
            # Draw all raw contours in thin blue lines to see exactly what cv2 found
            cv2.drawContours(debug_image, contours, -1, (255, 100, 0), 1)
        
        square_contours = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            centroid = None
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            def mark_fail(reason, color=(0, 0, 255)): # Default Red for fail
                if debug_image is not None and centroid:
                    cv2.putText(debug_image, reason, (centroid[0], centroid[1]), 
                               cv2.FONT_HERSHEY_PLAIN, 0.6, color, 1)

            # 1. Check area range
            if not (self.anchor_size_range[0] <= area <= self.anchor_size_range[1]):
                if 50 < area < 30000: mark_fail(f"area:{int(area)}")
                continue
            
            # 2. Check relative area (not too big)
            if area > (img_area * 0.05):
                mark_fail("large")
                continue
            
            # 3. Check aspect ratio
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            aspect_ratio = w_box / float(h_box) if h_box > 0 else 0
            min_ratio = 1.0 - self.aspect_ratio_tolerance
            max_ratio = 1.0 + self.aspect_ratio_tolerance
            
            if not (min_ratio <= aspect_ratio <= max_ratio):
                mark_fail(f"AR:{aspect_ratio:.1f}")
                continue
                
            # 4. Check solidity
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / float(hull_area) if hull_area > 0 else 0
            if solidity < 0.8:
                mark_fail(f"S:{solidity:.1f}")
                continue
            
            # 5. Check Extent
            extent = area / float(w_box * h_box) if (w_box * h_box) > 0 else 0
            if extent < 0.7: 
                mark_fail(f"E:{extent:.1f}")
                continue

            # 6. Check polygon approximation (RELAXED)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)
            if len(approx) < 4:
                mark_fail(f"v:{len(approx)}")
                continue
            
            # If passed all
            square_contours.append(cnt)
            if debug_image is not None and centroid:
                cv2.putText(debug_image, "PASS", (centroid[0], centroid[1]), 
                           cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 255, 0), 1)
        
        return square_contours
    
    def sort_corners(self, corners: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Identify the 4 extreme extremes (TL, TR, BR, BL) from a list of 4 corner points.
        Returns: List of 4 corner points in order [TL, TR, BR, BL]
        """
        points = np.array(corners, dtype="int32")
        rect = np.zeros((4, 2), dtype="int32")
        
        # Top-left has smallest sum, Bottom-right has largest sum
        s = points.sum(axis=1)
        rect[0] = points[np.argmin(s)]
        rect[2] = points[np.argmax(s)]
        
        # Top-right has smallest difference, Bottom-left has largest difference
        diff = np.diff(points, axis=1)
        rect[1] = points[np.argmin(diff)]
        rect[3] = points[np.argmax(diff)]
        
        return [tuple(pt) for pt in rect]
    
    def infer_fourth_anchor(self, corners: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        If only 3 anchors detected, infer the 4th using vector math.
        
        Assume missing corner. Use other 3 to compute 4th.
        Vector logic: If A=corner1, B=corner2, C=corner3, then D = C + (A - B)
        """
        if len(corners) == 4:
            return corners
        
        if len(corners) != 3:
            return None
        
        # Try to infer which corner is missing and compute it
        # Assume corners are roughly TL, TR, BR, BL
        # If we have 3, compute the missing one
        
        # Sort by position to figure out which is missing
        corners = sorted(corners, key=lambda p: (p[0], p[1]))
        
        # Simple heuristic: compute centroid and reflect
        cx = sum(c[0] for c in corners) / 3
        cy = sum(c[1] for c in corners) / 3
        
        # Find which quadrant each is in relative to centroid
        quadrants = []
        for c in corners:
            qx = 0 if c[0] < cx else 1
            qy = 0 if c[1] < cy else 1
            quadrants.append((qx, qy))
        
        # Find missing quadrant
        all_quadrants = {(0, 0), (0, 1), (1, 0), (1, 1)}
        present = set(quadrants)
        missing = list(all_quadrants - present)
        
        if not missing:
            return None
        
        missing_qx, missing_qy = missing[0]
        
        # Compute missing point by vector sum
        # Sum of all corners should form a rectangle
        sum_x = sum(c[0] for c in corners)
        sum_y = sum(c[1] for c in corners)
        
        computed_x = 2 * cx - sum_x / 3
        computed_y = 2 * cy - sum_y / 3
        
        corners.append((int(computed_x), int(computed_y)))
        return corners
    
    def crop_to_page(self, image: np.ndarray, margin: int = 50) -> np.ndarray:
        """
        Sơ bộ tìm vùng chứa tờ giấy và crop bớt nhiễu bên ngoài (bàn phím, viền bàn...).
        Sử dụng margin lớn (200px) để đảm bảo không cắt mất Anchor do phối cảnh nghiêng.
        Sau đó Scale up để giữ vững độ phân giải cho tờ giấy và điểm Neo.
        """
        h, w = image.shape[:2]
        
        # 1. Chuyển sang Gray và làm mờ mạnh để xóa chi tiết nhỏ
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        
        # 2. Otsu threshold để tách vùng sáng (giấy) và vùng tối (nền)
        # Sử dụng THRESH_BINARY vì giấy thường sáng hơn nền
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 3. Tìm contour lớn nhất (thường là tờ giấy)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image
            
        page_cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(page_cnt)
        
        # Nếu contour quá nhỏ so với ảnh (không phải tờ giấy), trả về ảnh gốc
        if area < (h * w * 0.1):
            return image
            
        # 4. Lấy bounding box và thêm margin
        x, y, bw, bh = cv2.boundingRect(page_cnt)
        
        x_start = max(0, x - margin)
        y_start = max(0, y - margin)
        x_end = min(w, x + bw + margin)
        y_end = min(h, y + bh + margin)
        
        cropped = image[y_start:y_end, x_start:x_end]
        
        # 5. Phóng to (Scale) lại một mức độ tương đương ảnh ban đầu để khôi phục độ phân giải
        # Dùng uniform scale (giữ nguyên tỉ lệ) để các ô vuông không bị méo lệch thành HCN
        scale = max(w / cropped.shape[1], h / cropped.shape[0])
        new_w = int(cropped.shape[1] * scale)
        new_h = int(cropped.shape[0] * scale)
        
        scaled_cropped = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return scaled_cropped

    def detect_anchors(self, image: np.ndarray) -> Optional[Tuple[List[Tuple[int, int]], np.ndarray]]:
        """
        Main detection pipeline: Crop -> Preprocess -> Find contours -> Extract corners -> Infer if needed.
        
        Returns: (anchor_corners, debug_image) or None if detection fails
        """
        # --- NEW: CROP TO PAGE FIRST ---
        # Giảm nhiễu từ môi trường bên ngoài (bàn phím, vật dụng trên bàn)
        # image_cropped = self.crop_to_page(image)
        image_cropped = image.copy() # Tạm thời bỏ crop để debug các ảnh khó, sau này sẽ bật lại khi logic crop đã ổn định hơn

        # Preprocess trên ảnh đã crop
        thresh = self.preprocess_image(image_cropped)
        
        # --- DEBUG OUTPUT IMPROVEMENT ---
        debug_img = image_cropped.copy()
        
        # 1. Find the strict, perfect contours
        strict_contours = self.find_square_contours(thresh, debug_image=debug_img)
        
        # Draw all candidates for debug
        all_raw_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in all_raw_contours:
            area = cv2.contourArea(cnt)
            if 50 < area < 30000: # Broad range for debug
                cv2.drawContours(debug_img, [cnt], -1, (255, 0, 0), 1) # Blue: Potential candidates

        # Extract all raw centroids from strict_contours
        centers = []
        for cnt in strict_contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                centers.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))
            else:
                x_b, y_b, w_b, h_b = cv2.boundingRect(cnt)
                centers.append((x_b + w_b // 2, y_b + h_b // 2))

        # --- FIX 2: THE RESCUE LOGIC ---
        if len(centers) == 3:
            print("  ⚠ 3 strict anchors found. Attempting to rescue the 4th...")
            
            # Guess where the 4th one is
            inferred_corners = self.infer_fourth_anchor(centers.copy())
            if inferred_corners and len(inferred_corners) == 4:
                missing_pt = inferred_corners[-1] # The newly appended guessed point
                
                rescued_cnt = None
                min_dist = float('inf')
                
                # Look through ALL raw contours to see if one is sitting on our guessed point
                for cnt in all_raw_contours:
                    area = cv2.contourArea(cnt)
                    if area < 20 or area > 30000: # Mở rộng khoảng kiểm tra cho ảnh rất xa
                        continue
                        
                    # Calculate centroid of this raw contour
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # Distance formula to see how close it is to our guess
                        dist = np.sqrt((cx - missing_pt[0])**2 + (cy - missing_pt[1])**2)
                        
                        # If it's within a reasonable radius (e.g., 200 pixels) of our guess
                        if dist < 200 and dist < min_dist:
                            min_dist = dist
                            rescued_cnt = cnt
                            
                if rescued_cnt is not None:
                    strict_contours.append(rescued_cnt)
                    # Add to our centers list explicitly
                    M = cv2.moments(rescued_cnt)
                    if M["m00"] != 0:
                        centers.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))
                    print(f"  ✓ Successfully rescued contour at distance {int(min_dist)}px")
                    cv2.putText(debug_img, "RESCUED", (int(missing_pt[0]), int(missing_pt[1])), 
                               cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 165, 255), 2) # Orange text

        # Draw SELECTED candidates (yellow)
        cv2.drawContours(debug_img, strict_contours, -1, (0, 255, 255), 3) 
        
        if len(centers) < 3:
            print(f"  ⚠ Only {len(centers)} valid square anchors detected (need ≥3)")
            # Still return debug image so user can see what was detected
            return None, debug_img
            
        # Infer 4th corner if *still* only 3 detected (Rescue logic visually failed)
        if len(centers) == 3:
            print(f"  ⚠ Only {len(centers)} corners found visually, falling back to math inference...")
            corners = self.infer_fourth_anchor(centers)
        else:
            corners = centers
        
        if corners is None or len(corners) < 4:
            print("  ❌ Failed to detect/infer 4 anchors")
            return None, debug_img
            
        # Sort the final 4 corners
        corners = self.sort_corners(corners[:4])
        
        # Mark final 4 corners (green)
        for i, corner in enumerate(corners):
            cv2.circle(debug_img, corner, 15, (0, 255, 0), 3)
            cv2.putText(debug_img, f"Anchor {i}", (corner[0] + 20, corner[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return corners, debug_img
    
    def warp_perspective(self, image: np.ndarray, src_corners: List[Tuple[int, int]], 
                         output_size: Tuple[int, int] = (2100, 2970)) -> Tuple[np.ndarray, np.ndarray]:
        """
        Warp image to rectangular form using detected anchors.
        
        Args:
            image: Original image
            src_corners: Detected anchor corners [TL, TR, BR, BL]
            output_size: (width, height) of output warped image
            
        Returns: (warped_image, transformation_matrix)
        """
        # Define destination corners (perfect rectangle)
        dst_corners = np.float32([
            [0, 0],                              # Top-left
            [output_size[0], 0],                # Top-right
            [output_size[0], output_size[1]],   # Bottom-right
            [0, output_size[1]]                 # Bottom-left
        ])
        
        src_corners = np.float32(src_corners)
        
        # Compute perspective transformation matrix
        M = cv2.getPerspectiveTransform(src_corners, dst_corners)
        
        # Warp image
        warped = cv2.warpPerspective(image, M, output_size)
        
        return warped, M


# Test the scanner
if __name__ == "__main__":
    import os
    from pathlib import Path
    
    images_dir = "test/images/ver2"
    output_dir = "outputs/scanner_debug_ver2"
    os.makedirs(output_dir, exist_ok=True)
    
    # Try multiple configurations for stubborn images
    scanner_configs = [
        {"anchor_size_range": (100, 8000), "aspect_ratio_tolerance": 0.3},   # Standard
        {"anchor_size_range": (20, 15000), "aspect_ratio_tolerance": 0.5},   # Super lenient for far/tiny anchors
    ]
    
    # Process all images
    for img_file in sorted(os.listdir(images_dir)):
        if Path(img_file).suffix.lower() not in {".jpg", ".png", ".jpeg", ".bmp"}:
            continue
        
        img_path = os.path.join(images_dir, img_file)
        print(f"\n📷 Processing: {img_file}")
        
        img = cv2.imread(img_path)
        if img is None:
            print("  ❌ Failed to load")
            continue
        
        result = None
        
        # Try different configurations
        result_anchors = None
        final_corners = None
        final_debug_img = None
        
        for config_idx, config in enumerate(scanner_configs):
            scanner = RobustOMRScanner(**config)
            result_anchors = scanner.detect_anchors(img)
            
            if result_anchors:
                final_corners, final_debug_img = result_anchors
                
                if final_corners is not None:
                    print(f"  ✓ Detected with config {config_idx}: {config}")
                    break
        
        if final_corners is not None:
            print(f"  ✓ Detected 4 anchors:")
            for i, c in enumerate(final_corners):
                print(f"    Corner {i}: {c}")
            
            # Save debug image
            debug_path = os.path.join(output_dir, f"{Path(img_file).stem}_anchors.png")
            cv2.imwrite(debug_path, final_debug_img)
            
            # Warp perspective
            warped, M = scanner.warp_perspective(img, final_corners)
            warped_path = os.path.join(output_dir, f"{Path(img_file).stem}_warped.png")
            cv2.imwrite(warped_path, warped)
            print(f"  ✓ Saved debug: {debug_path}")
            print(f"  ✓ Saved warped: {warped_path}")
        else:
            # SAVE DEBUG IMAGE EVEN IF FAILED
            if final_debug_img is not None:
                debug_path = os.path.join(output_dir, f"{Path(img_file).stem}_FAILED_anchors.png")
                cv2.imwrite(debug_path, final_debug_img)
                print(f"  ⚠ Saved FAILED debug image: {debug_path}")
            
            print("  ❌ Failed to detect anchors with all configs")
    
    print(f"\n✓ All results saved to {output_dir}/")
