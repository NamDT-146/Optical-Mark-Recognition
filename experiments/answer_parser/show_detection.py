import os
import sys
import cv2
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from answer_parser import AnswerParser


def main():
    img_path = os.path.join(script_dir, "normal_warped_debug.png")
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return

    out_bin = os.path.join(script_dir, "normal_warped_binarized.png")
    out_circles = os.path.join(script_dir, "normal_warped_circles.png")

    parser = AnswerParser(os.path.join(script_dir, os.pardir, os.pardir, "test", "templates", "Professional_OMR_45.json"))

    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to read image: {img_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use the parser's local binarization (works for full image too)
    bin_img = parser._binarize_crop(gray)
    # Save binarized image (invert to normal white background for easier viewing)
    bin_save = cv2.bitwise_not(bin_img)
    cv2.imwrite(out_bin, bin_save)

    # Contour-based circle detection
    overlay = img.copy()
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cont_count = 0
    h, w = gray.shape[:2]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 30:
            continue
        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        if r < 4 or r > max(w, h) * 0.1:
            continue
        circularity = (4.0 * np.pi * area) / (peri * peri)
        if circularity < 0.45:
            continue
        cont_count += 1
        cv2.circle(overlay, (int(round(cx)), int(round(cy))), int(round(r)), (0, 255, 0), 2)

    # HoughCircles fallback / additional detection
    blur = cv2.GaussianBlur(gray, (9, 9), 2)
    try:
        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, param1=100, param2=20, minRadius=4, maxRadius=120)
    except Exception:
        circles = None

    hough_count = 0
    if circles is not None:
        for c in circles[0]:
            cx, cy, r = int(round(c[0])), int(round(c[1])), int(round(c[2]))
            cv2.circle(overlay, (cx, cy), r, (0, 0, 255), 2)
            cv2.circle(overlay, (cx, cy), 2, (0, 0, 255), -1)
            hough_count += 1

    # Annotate counts and save
    cv2.putText(overlay, f"Contours: {cont_count}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(overlay, f"Hough: {hough_count}", (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.imwrite(out_circles, overlay)

    print(f"Binarized image saved to: {out_bin}")
    print(f"Circle-detection overlay saved to: {out_circles}")


if __name__ == '__main__':
    main()
