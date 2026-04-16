import cv2
import numpy as np

def test_grid_strategy(image_path):
    img = cv2.imread(image_path)
    if img is None: return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    out_img = img.copy()
    h_img, w_img = binary.shape[:2]
    
    column_boxes = []
    
    # Target box ratio around 2.6 (115/42) for 20-row columns
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area > h_img * w_img * 0.01: 
            ratio = h / max(1, w)
            if 2.4 <= ratio <= 2.8 and y > h_img * 0.4:
                column_boxes.append((x, y, w, h))

    column_boxes = sorted(column_boxes, key=lambda b: b[0])
    
    for i, (bx, by, bw, bh) in enumerate(column_boxes):
        cv2.rectangle(out_img, (bx, by), (bx+bw, by+bh), (0, 255, 255), 2)
        
        # Draw expected centers
        for r in range(20): # 20 rows per column
            cy_pct = (4.0 + r * 5.5) / 115.0
            cy = by + int(cy_pct * bh)
            
            for j in range(4): # 4 options
                cx_pct = (10.0 + j * 8.0) / 42.0
                cx = bx + int(cx_pct * bw)
                
                # Draw center
                cv2.circle(out_img, (cx, cy), 3, (0, 0, 255), -1)
                
                # Draw cell bounding box around it (approx 8mm x 5.5mm cell)
                cell_w = int((8.0 / 42.0) * bw)
                cell_h = int((5.5 / 115.0) * bh)
                cv2.rectangle(out_img, (cx - cell_w//2, cy - cell_h//2), (cx + cell_w//2, cy + cell_h//2), (0, 255, 0), 1)

    cv2.imwrite("experiments/answer_parser/grid_boxes.png", out_img)

if __name__ == '__main__':
    test_grid_strategy("outputs/scanner_debug_ver2/normal_warped.png")
