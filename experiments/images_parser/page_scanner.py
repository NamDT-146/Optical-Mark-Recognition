import cv2
import numpy as np
from typing import Optional, Tuple, List


class PageBasedOMRScanner:
    """
    Smart OMR scanner using page/paper detection instead of anchor marks.
    Uses center of image as seed point to find paper boundary.
    
    Much more robust for real-world conditions:
    - B&W printed documents
    - Missing/damaged anchors
    - Uneven lighting and shadows
    - Documents placed on dark backgrounds
    """
    
    def __init__(
        self,
        min_paper_area: int = 10000,
        min_contour_darkness_ratio: float = 0.3,
        min_paper_ratio_area: float = 0.15,
        ransac_iterations: int = 500,
        use_anchor_refinement: bool = True,
    ):
        """
        Args:
            min_paper_area: Minimum area (in pixels) to be considered as paper
            min_contour_darkness_ratio: Ratio of dark pixels in contour to confirm validity
            min_paper_ratio_area: Minimum paper area ratio compared to full image
            ransac_iterations: Number of RANSAC hypotheses for rotated rectangle fitting
            use_anchor_refinement: Whether to refine detected corners with local anchor search
        """
        self.min_paper_area = min_paper_area
        self.min_contour_darkness_ratio = min_contour_darkness_ratio
        self.min_paper_ratio_area = min_paper_ratio_area
        self.ransac_iterations = ransac_iterations
        self.use_anchor_refinement = use_anchor_refinement
    
    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocessing: Grayscale + Denoising + Adaptive Threshold.
        Returns both the threshold and the enhanced grayscale.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
        
        # Gaussian blur
        blurred = cv2.GaussianBlur(denoised, (5, 5), 0)
        
        # CLAHE for shadow handling
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, blockSize=21, C=5
        )
        
        # Morphological closing to fill small holes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        return thresh, enhanced
    
    def sample_center_color(self, image: np.ndarray) -> Tuple[int, int, int]:
        """
        Sample the color at the center of the image to determine 'paper color'.
        For B&W documents on dark background, this should be light colored.
        """
        h, w = image.shape[:2]
        center_y, center_x = h // 2, w // 2
        
        # Use a small region around center for more robust sampling
        region_size = 50
        center_region = image[
            max(0, center_y - region_size):min(h, center_y + region_size),
            max(0, center_x - region_size):min(w, center_x + region_size)
        ]
        
        # Get median color of center region
        b, g, r = [np.median(center_region[:,:,i]) for i in range(3)]
        return int(r), int(g), int(b)
    
    def find_paper_contour(self, thresh: np.ndarray, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Find the contour representing the paper/document.
        Strategy: Find the largest contour that contains the image center and is >= 15% of image area.
        
        Returns: The contour representing the paper boundary
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        image_area = h * w
        min_paper_ratio_area = image_area * self.min_paper_ratio_area
        
        # Find all contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("⚠ No contours found in threshold image")
            return None
        
        # Filter contours that:
        # 1. Contain the center point
        # 2. Have sufficient area (minimum AND at least 15% of image)
        # 3. Are roughly rectangular
        candidate_contours = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            # Minimum area check
            if area < self.min_paper_area:
                continue
            
            # Area ratio check - contour must be at least 15% of original image
            if area < min_paper_ratio_area:
                print(
                    f"  ⚠ Contour area {area:.0f}px² is only {100*area/image_area:.1f}% "
                    f"(need ≥{100*self.min_paper_ratio_area:.0f}%)"
                )
                continue
            
            # Check if center is inside this contour
            dist = cv2.pointPolygonTest(cnt, center, False)
            if dist < 0:  # Point is outside
                continue
            
            # Check if roughly rectangular (4-sided or can be approximated to 4 sides)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            
            # Accept if it has 3-6 vertices (rough rectangle)
            if len(approx) >= 3:
                candidate_contours.append((area, cnt))
        
        if not candidate_contours:
            print(
                "  ⚠ No valid paper contours found "
                f"(must be >={100*self.min_paper_ratio_area:.0f}% of image area)"
            )
            return None
        
        # Sort by area (descending) and return the largest
        candidate_contours.sort(key=lambda x: x[0], reverse=True)
        paper_contour = candidate_contours[0][1]
        
        return paper_contour
    
    def extract_corners(self, contour: np.ndarray, thresh: np.ndarray) -> Optional[List[Tuple[int, int]]]:
        """
        Extract 4 corner points from paper contour.
        
        Process:
        1. Approximate contour to polygon (orientation seeds)
        2. Fit rotated rectangle model with RANSAC-style angle hypotheses
        3. Convert best model to 4 corners [TL, TR, BR, BL]
        
        Returns: [TL, TR, BR, BL] corners or None if detection fails
        """
        # 1) Approximate contour to a polygon for stable orientation seeds.
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.005 * peri, True)

        approx_points = approx.reshape(-1, 2)
        if len(approx_points) < 4:
            print(f"⚠ Contour has only {len(approx_points)} vertices (need ≥4)")
            return None

        all_points = contour.reshape(-1, 2).astype(np.float32)
        corners = self._fit_rotated_rectangle_ransac(
            contour_points=all_points,
            seed_points=approx_points.astype(np.float32),
            image_shape=thresh.shape,
            thresh=thresh,
        )
        if corners is not None:
            return corners

        return self._fallback_corners_from_min_area_rect(contour)

    @staticmethod
    def _normalize_angle_pi(theta: float) -> float:
        """Normalize angle to [0, pi)."""
        return float(theta % np.pi)

    def _collect_candidate_angles(self, seed_points: np.ndarray) -> List[float]:
        """
        Build candidate rectangle orientations.

        Uses polygon edges first, then random pairs (RANSAC) for robustness.
        """
        angles: List[float] = []

        n = len(seed_points)
        if n >= 2:
            for i in range(n):
                p1 = seed_points[i]
                p2 = seed_points[(i + 1) % n]
                dx, dy = p2 - p1
                edge_len = float(np.hypot(dx, dy))
                if edge_len < 5:
                    continue
                theta = self._normalize_angle_pi(np.arctan2(float(dy), float(dx)))
                angles.append(theta)
                angles.append(self._normalize_angle_pi(theta + np.pi / 2.0))

        if n >= 2:
            for _ in range(self.ransac_iterations):
                idx1, idx2 = np.random.choice(n, 2, replace=False)
                p1 = seed_points[idx1]
                p2 = seed_points[idx2]
                dx, dy = p2 - p1
                if float(np.hypot(dx, dy)) < 5:
                    continue
                theta = self._normalize_angle_pi(np.arctan2(float(dy), float(dx)))
                angles.append(theta)

        # Coarse grid helps when contour approximation misses dominant direction.
        for deg in range(0, 180, 10):
            angles.append(np.deg2rad(deg))

        # De-duplicate with small tolerance.
        angles = sorted(angles)
        unique_angles: List[float] = []
        min_sep = np.deg2rad(1.0)
        for theta in angles:
            if not unique_angles or abs(theta - unique_angles[-1]) > min_sep:
                unique_angles.append(theta)

        return unique_angles

    @staticmethod
    def _rectangle_from_theta(points: np.ndarray, theta: float, q: float = 0.02) -> Optional[Tuple[float, float, float, float, np.ndarray, np.ndarray]]:
        """
        Build rotated rectangle model from orientation theta.

        Formula:
        - u = (cos(theta), sin(theta)), v = (-sin(theta), cos(theta))
        - su = p dot u, sv = p dot v
        - Rectangle edges are su=u_min, su=u_max, sv=v_min, sv=v_max
        """
        u = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
        v = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float32)

        proj_u = points @ u
        proj_v = points @ v

        u_min = float(np.quantile(proj_u, q))
        u_max = float(np.quantile(proj_u, 1.0 - q))
        v_min = float(np.quantile(proj_v, q))
        v_max = float(np.quantile(proj_v, 1.0 - q))

        if u_max <= u_min or v_max <= v_min:
            return None

        return u_min, u_max, v_min, v_max, u, v

    @staticmethod
    def _corners_from_model(model: Tuple[float, float, float, float, np.ndarray, np.ndarray]) -> np.ndarray:
        """Convert rotated rectangle model to 4 corner points (float)."""
        u_min, u_max, v_min, v_max, u, v = model

        # Corners in local (u, v) rectangle coordinates.
        local_corners = np.array(
            [
                [u_min, v_min],
                [u_max, v_min],
                [u_max, v_max],
                [u_min, v_max],
            ],
            dtype=np.float32,
        )

        # p = su*u + sv*v
        corners = np.stack([
            local_corners[:, 0] * u[0] + local_corners[:, 1] * v[0],
            local_corners[:, 0] * u[1] + local_corners[:, 1] * v[1],
        ], axis=1)
        return corners

    def _anchor_evidence_score(self, thresh: np.ndarray, corners: np.ndarray) -> float:
        """Estimate anchor evidence from binary mask around candidate corners."""
        h, w = thresh.shape[:2]
        radius = max(10, int(0.025 * min(h, w)))

        scores: List[float] = []
        for x_f, y_f in corners:
            x = int(round(float(x_f)))
            y = int(round(float(y_f)))
            x1, x2 = max(0, x - radius), min(w, x + radius)
            y1, y2 = max(0, y - radius), min(h, y + radius)
            if x2 <= x1 or y2 <= y1:
                continue

            patch = thresh[y1:y2, x1:x2]
            if patch.size == 0:
                continue

            # thresh is binary_inv: white pixels represent dark ink/marks.
            ink_ratio = float(np.count_nonzero(patch)) / float(patch.size)
            # Reward moderate/high ink density but clamp to avoid overfitting noise.
            scores.append(min(ink_ratio / 0.40, 1.0))

        if not scores:
            return 0.0
        return float(np.mean(scores))

    def _score_rectangle_model(
        self,
        points: np.ndarray,
        model: Tuple[float, float, float, float, np.ndarray, np.ndarray],
        image_shape: Tuple[int, int],
        thresh: Optional[np.ndarray] = None,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Score rectangle candidate.

        Priority:
        1) Pass through many contour points
        2) Prefer longer edges when scores are close
        3) Small optional bonus for corner-anchor evidence
        """
        h, w = image_shape
        u_min, u_max, v_min, v_max, u, v = model

        proj_u = points @ u
        proj_v = points @ v

        d_edge = np.minimum.reduce([
            np.abs(proj_u - u_min),
            np.abs(proj_u - u_max),
            np.abs(proj_v - v_min),
            np.abs(proj_v - v_max),
        ])

        inlier_threshold = max(4.0, 0.012 * float(np.hypot(h, w)))
        inlier_mask = d_edge < inlier_threshold
        inlier_count = float(np.count_nonzero(inlier_mask))
        inlier_ratio = inlier_count / max(float(len(points)), 1.0)

        width = u_max - u_min
        height = v_max - v_min
        perimeter = 2.0 * (width + height)
        rect_area = width * height

        perimeter_norm = perimeter / max(2.0 * (h + w), 1.0)
        area_norm = rect_area / max(float(h * w), 1.0)

        corners = self._corners_from_model(model)
        anchor_bonus = 0.0
        if thresh is not None:
            anchor_bonus = self._anchor_evidence_score(thresh, corners)

        score = (
            0.68 * inlier_ratio +
            0.20 * perimeter_norm +
            0.07 * area_norm +
            0.05 * anchor_bonus
        )

        return float(score), inlier_mask, corners

    def _fit_rotated_rectangle_ransac(
        self,
        contour_points: np.ndarray,
        seed_points: np.ndarray,
        image_shape: Tuple[int, int],
        thresh: Optional[np.ndarray] = None,
    ) -> Optional[List[Tuple[int, int]]]:
        """
        RANSAC-style rotated rectangle fitting.

        Rectangle model is parameterized by orientation (theta) and 4 projected limits:
        su = p dot u, sv = p dot v
        edges: su=u_min, su=u_max, sv=v_min, sv=v_max
        """
        h, w = image_shape
        candidate_angles = self._collect_candidate_angles(seed_points)
        if not candidate_angles:
            return None

        best_score = -1.0
        best_corners: Optional[np.ndarray] = None

        for theta in candidate_angles:
            model = self._rectangle_from_theta(contour_points, theta)
            if model is None:
                continue

            score, _, corners = self._score_rectangle_model(contour_points, model, (h, w), thresh=thresh)
            if score > best_score:
                best_score = score
                best_corners = corners

        if best_corners is None:
            return None

        corners_int = [
            (int(round(float(x))), int(round(float(y))))
            for x, y in best_corners
        ]
        return self._sort_corners_by_position(corners_int)

    def _fallback_corners_from_min_area_rect(self, contour: np.ndarray) -> List[Tuple[int, int]]:
        """Fallback corner extraction using OpenCV minAreaRect."""
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        corners = [
            (int(round(float(p[0]))), int(round(float(p[1]))))
            for p in box
        ]
        return self._sort_corners_by_position(corners)
    
    def _fallback_corners(self, points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Fallback: get Top 4 extreme points based on distance to centroid"""
        cx = np.mean([p[0] for p in points])
        cy = np.mean([p[1] for p in points])
        scored = []
        for pt in points:
            dist = (pt[0] - cx)**2 + (pt[1] - cy)**2
            scored.append((dist, tuple(pt)))
        scored.sort(key=lambda x: x[0], reverse=True)
        return self._sort_corners_by_position([s[1] for s in scored[:4]])

    
    def _sort_corners_by_position(self, corners: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Sort corners into [TL, TR, BR, BL] order using position-based scoring.
        """
        if len(corners) != 4:
            return corners
        
        # Sort by position (sum and difference)
        scored = [(p, p[0] + p[1], p[0] - p[1]) for p in corners]
        scored.sort(key=lambda x: (x[1], x[2]))
        
        # TL = smallest sum (top-left)
        # BR = largest sum (bottom-right)
        # Middle two: sort by difference (x-y)
        tl = scored[0][0]
        br = scored[3][0]
        
        # Middle two: sort by difference
        middle = sorted([scored[1], scored[2]], key=lambda x: x[2])
        bl = middle[0][0]   # Small difference = bottom-left
        tr = middle[1][0]   # Large difference = top-right
        
        return [tl, tr, br, bl]
    
    def _extract_edge_points(self, contour: np.ndarray) -> List[Tuple[int, int]]:
        """
        Extract edge points from contour.
        Returns list of (x, y) coordinates representing the paper edge.
        """
        edge_points = []
        
        # Get all points from the contour
        for point in contour:
            x, y = point[0]
            edge_points.append((int(x), int(y)))
        
        return edge_points

    def _find_anchor_near_corner(
        self,
        corner: Tuple[int, int],
        thresh: np.ndarray,
    ) -> Optional[Tuple[int, int]]:
        """
        Find a dark/square anchor near one corner and return its outermost point.

        Returning the outermost contour point (away from image center) better aligns with
        page corner than anchor centroid when anchor is printed near a corner.
        """
        h, w = thresh.shape[:2]
        cx, cy = corner
        radius = max(25, int(0.08 * min(h, w)))

        x1, x2 = max(0, cx - radius), min(w, cx + radius)
        y1, y2 = max(0, cy - radius), min(h, cy + radius)
        if x2 <= x1 or y2 <= y1:
            return None

        local = thresh[y1:y2, x1:x2]
        local_contours, _ = cv2.findContours(local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not local_contours:
            return None

        window_area = float((x2 - x1) * (y2 - y1))
        min_area = 0.004 * window_area
        max_area = 0.40 * window_area

        center_local = np.array([cx - x1, cy - y1], dtype=np.float32)
        best_score = -1.0
        best_cnt_global = None

        for cnt in local_contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            rect = cv2.minAreaRect(cnt)
            rw, rh = rect[1]
            if rw < 2 or rh < 2:
                continue

            aspect = min(rw, rh) / max(rw, rh)
            if aspect < 0.50:
                continue

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area <= 0:
                continue
            solidity = area / hull_area
            if solidity < 0.65:
                continue

            mask = np.zeros(local.shape, dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            ink_ratio = (cv2.mean(local, mask=mask)[0] / 255.0)

            m = cv2.moments(cnt)
            if m["m00"] > 0:
                c_local = np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]], dtype=np.float32)
            else:
                x, y, ww, hh = cv2.boundingRect(cnt)
                c_local = np.array([x + ww / 2.0, y + hh / 2.0], dtype=np.float32)

            dist = float(np.linalg.norm(c_local - center_local))
            dist_score = max(0.0, 1.0 - dist / (radius * 1.4))

            # Prefer larger/squarer/ink-richer components close to predicted corner.
            area_score = min(area / max_area, 1.0)
            score = (
                0.30 * area_score +
                0.25 * aspect +
                0.20 * solidity +
                0.15 * ink_ratio +
                0.10 * dist_score
            )

            if score > best_score:
                best_score = score
                cnt_global = cnt + np.array([[[x1, y1]]], dtype=np.int32)
                best_cnt_global = cnt_global

        if best_cnt_global is None or best_score < 0.45:
            return None

        img_center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
        pts = best_cnt_global.reshape(-1, 2).astype(np.float32)
        d2 = np.sum((pts - img_center) ** 2, axis=1)
        best_idx = int(np.argmax(d2))
        best_pt = pts[best_idx]
        return int(round(float(best_pt[0]))), int(round(float(best_pt[1])))

    def refine_corners_with_anchors(
        self,
        corners: List[Tuple[int, int]],
        thresh: np.ndarray,
    ) -> Tuple[List[Tuple[int, int]], int]:
        """Refine page corners with nearby dark-square anchors if available."""
        refined = list(corners)
        found = 0

        for i, corner in enumerate(corners):
            anchor_corner = self._find_anchor_near_corner(corner, thresh)
            if anchor_corner is not None:
                refined[i] = anchor_corner
                found += 1

        return self._sort_corners_by_position(refined), found

    
    def detect_page(self, image: np.ndarray) -> Optional[Tuple[List[Tuple[int, int]], np.ndarray]]:
        """
        Main detection pipeline: Find paper boundary and extract corners.
        
        Improvements:
        - Validates contour area is >= 15% of image
        - Extracts corners with rotated-rectangle RANSAC model
        - Prefers larger edge models when candidates are similar
        - Optionally refines corners with local anchor detection
        - Verifies corners form valid quadrilateral
        
        Returns: (corners, debug_image) or None if detection fails
        """
        print(f"Processing image: {image.shape}")
        
        # Preprocess
        thresh, _ = self.preprocess_image(image)
        
        # Find paper contour
        paper_contour = self.find_paper_contour(thresh, image)
        
        if paper_contour is None:
            print("❌ Failed to detect paper contour")
            return None
        
        # Extract corners
        corners = self.extract_corners(paper_contour, thresh)
        
        if corners is None or len(corners) < 4:
            print("❌ Failed to extract 4 corners")
            return None
        
        # Validate corners form reasonable quadrilateral
        if not self._validate_corners(corners, image.shape[:2]):
            print("❌ Detected corners do not form valid page boundary")
            return None

        # Optional refinement with local dark anchors near 4 corners.
        if self.use_anchor_refinement:
            refined_corners, anchor_count = self.refine_corners_with_anchors(corners, thresh)
            if anchor_count > 0 and self._validate_corners(refined_corners, image.shape[:2]):
                print(f"  ✓ Anchor refinement applied ({anchor_count} corner(s) updated)")
                corners = refined_corners
        
        print(f"✓ Detected paper with 4 corners:")
        for i, corner in enumerate(corners):
            print(f"  Corner {i}: {corner}")
        
        # Create debug image with yellow edge points and green corners
        debug_img = image.copy()
        
        # Draw paper contour
        cv2.drawContours(debug_img, [paper_contour], 0, (0, 255, 255), 3)
        
        # Draw edge points in yellow
        for i in range(len(paper_contour)):
            pt = tuple(paper_contour[i][0])
            cv2.circle(debug_img, pt, 3, (0, 255, 255), -1)  # Yellow points
        
        # Draw corners in green
        for i, corner in enumerate(corners):
            cv2.circle(debug_img, corner, 15, (0, 255, 0), 3)  # Green circles
            cv2.putText(debug_img, str(i), (corner[0] + 20, corner[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw center point
        center = (image.shape[1] // 2, image.shape[0] // 2)
        cv2.circle(debug_img, center, 10, (255, 0, 0), 2)
        
        return corners, debug_img
    
    def _validate_corners(self, corners: List[Tuple[int, int]], image_shape: Tuple[int, int]) -> bool:
        """
        Validate that corners form a reasonable quadrilateral.
        
        Checks:
        1. Corners are distinct (not too close to each other)
        2. Corners form a reasonable spread
        3. Corner distances are realistic
        
        Returns: True if valid, False otherwise
        """
        if len(corners) != 4:
            return False
        
        h, w = image_shape
        
        # Check 1: Corners are distinct (at least 15px apart to allow for extreme angles)
        min_distance = 15
        for i in range(len(corners)):
            for j in range(i+1, len(corners)):
                dist = np.sqrt((corners[i][0] - corners[j][0])**2 + 
                              (corners[i][1] - corners[j][1])**2)
                if dist < min_distance:
                    print(f"  ⚠ Corners {i} and {j} too close ({dist:.0f}px)")
                    return False
        
        # Check 2: Corners should roughly span the image (at least 30% spread)
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        
        x_range = max(xs) - min(xs)
        y_range = max(ys) - min(ys)
        
        # Check that corners span at least 30% of image dimensions (relaxed from 40%)
        min_x_span = 0.30
        min_y_span = 0.30
        
        if x_range < min_x_span * w:
            print(f"  ⚠ Corners x-range {x_range:.0f}px < {min_x_span*w:.0f}px (30% of width)")
            return False
        
        if y_range < min_y_span * h:
            print(f"  ⚠ Corners y-range {y_range:.0f}px < {min_y_span*h:.0f}px (30% of height)")
            return False
        
        # Check 3: Corners should be reasonably within image bounds (allow slight overdraw)
        margin = 50  # Allow some overdraw for extreme angles
        for i, (x, y) in enumerate(corners):
            if not (-margin <= x <= w+margin and -margin <= y <= h+margin):
                print(f"  ⚠ Corner {i} ({x},{y}) way outside image ({w}×{h})")
                return False
        
        print(f"  ✓ Corners validation passed (span: {x_range:.0f}×{y_range:.0f}px, {100*x_range/w:.1f}%×{100*y_range/h:.1f}%)")
        return True
    
    def warp_perspective(self, image: np.ndarray, corners: List[Tuple[int, int]],
                         output_size: Tuple[int, int] = (2100, 2970)) -> Tuple[np.ndarray, np.ndarray]:
        """
        Warp image to rectangular form using detected corners.
        
        Returns: (warped_image, transformation_matrix)
        """
        # Destination corners (perfect rectangle)
        dst_corners = np.float32([
            [0, 0],
            [output_size[0], 0],
            [output_size[0], output_size[1]],
            [0, output_size[1]]
        ])
        
        src_corners = np.float32(corners)
        
        # Compute perspective transformation
        M = cv2.getPerspectiveTransform(src_corners, dst_corners)
        
        # Warp
        warped = cv2.warpPerspective(image, M, output_size)
        
        return warped, M


# Test the page-based scanner
if __name__ == "__main__":
    import os
    from pathlib import Path
    
    scanner = PageBasedOMRScanner(min_paper_area=50000)
    
    images_dir = "test/images/ver2"
    output_dir = "outputs/page_scanner_debug_ver2"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== Page-Based OMR Scanner Test ===\n")
    
    for img_file in sorted(os.listdir(images_dir)):
        if Path(img_file).suffix.lower() not in {".jpg", ".png", ".jpeg", ".bmp"}:
            continue
        
        img_path = os.path.join(images_dir, img_file)
        print(f"📷 {img_file}")
        
        img = cv2.imread(img_path)
        if img is None:
            print("  ❌ Failed to load\n")
            continue
        
        # Detect page
        result = scanner.detect_page(img)
        
        if result:
            corners, debug_img = result
            
            # Save debug image
            debug_path = os.path.join(output_dir, f"{Path(img_file).stem}_page_detected.png")
            cv2.imwrite(debug_path, debug_img)
            
            # Warp perspective
            warped, M = scanner.warp_perspective(img, corners)
            warped_path = os.path.join(output_dir, f"{Path(img_file).stem}_warped.png")
            cv2.imwrite(warped_path, warped)
            
            print(f"  ✓ Warped size: {warped.shape[1]}x{warped.shape[0]}")
            print(f"  ✓ Saved: {debug_path}")
            print(f"  ✓ Saved: {warped_path}\n")
        else:
            print("  ❌ Detection failed\n")
    
    print(f"✓ Results saved to {output_dir}/")
