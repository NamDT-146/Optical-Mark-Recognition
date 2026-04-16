import cv2
import numpy as np

img = cv2.imread('outputs/scanner_debug_ver2/normal_warped.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
_, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

out = img.copy()
h_img, w_img = binary.shape[:2]

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if w * h > h_img * w_img * 0.001:
        ratio = h / max(1, w)
        if y < h_img * 0.4:
            if abs(ratio - 3.83) < 0.5:
                cv2.rectangle(out, (x, y), (x+w, y+h), (0, 0, 255), 3)
                print(f"Exam code box: x={x}, y={y}, w={w}, h={h}, ratio={ratio:.2f}")
            elif abs(ratio - 1.15) < 0.3:
                cv2.rectangle(out, (x, y), (x+w, y+h), (0, 255, 0), 3)
                print(f"Student ID box: x={x}, y={y}, w={w}, h={h}, ratio={ratio:.2f}")

cv2.imwrite('outputs/scanner_debug_ver2/test_info_boxes.png', out)
