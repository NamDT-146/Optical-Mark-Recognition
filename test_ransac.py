import cv2
import numpy as np

def ransac_4_lines_corners(contour):
    points = contour.reshape(-1, 2)
    num_lines = 4
    iterations = 500
    threshold = 5.0
    
    lines = []
    remaining_points = points.copy()
    
    for _ in range(num_lines):
        if len(remaining_points) < 2:
            break
        best_line = None
        max_inliers = 0
        best_inliers_idx = None
        for _ in range(iterations):
            idx1, idx2 = np.random.choice(len(remaining_points), 2, replace=False)
            p1, p2 = remaining_points[idx1], remaining_points[idx2]
            ux = p2[0] - p1[0]
            uy = p2[1] - p1[1]
            length = np.hypot(ux, uy)
            if length == 0: continue
            ux, uy = ux/length, uy/length
            nx, ny = -uy, ux
            
            diffs = remaining_points - p1
            dists = np.abs(diffs[:, 0] * nx + diffs[:, 1] * ny)
            inlier_indices = np.where(dists < threshold)[0]
            inlier_count = len(inlier_indices)
            if inlier_count > max_inliers:
                max_inliers = inlier_count
                best_inliers_idx = inlier_indices
                best_line = (p1, (ux, uy))
        if best_line is None: break
        
        inlier_points = remaining_points[best_inliers_idx]
        mean = np.mean(inlier_points, axis=0)
        centered = inlier_points - mean
        if len(inlier_points) >= 2:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            direction = vh[0]
        else:
            direction = best_line[1]
            
        lines.append((mean, direction))
        remaining_points = np.delete(remaining_points, best_inliers_idx, axis=0)
        
    intersections = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            p1, d1 = lines[i]
            p2, d2 = lines[j]
            cross = d1[0]*d2[1] - d1[1]*d2[0]
            if abs(cross) < 1e-5: continue
            t1 = ((p2[0] - p1[0])*d2[1] - (p2[1] - p1[1])*d2[0]) / cross
            inter_p = p1 + t1 * d1
            intersections.append((int(inter_p[0]), int(inter_p[1])))
    
    # Filter 4 corners closest to centroid of all points
    cx = np.mean(points[:, 0])
    cy = np.mean(points[:, 1])
    intersections.sort(key=lambda p: (p[0]-cx)**2 + (p[1]-cy)**2)
    return intersections[:4]

# dummy contour
cnt = np.array([[[10, 10]], [[10, 100]], [[100, 100]], [[100, 10]]])
print(ransac_4_lines_corners(cnt))
