import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

image_paths = [
    "test/images/flash_far.jpg",
    "test/images/flash_near.jpg",
    "test/images/shade.jpg",
    "test/images/too_close.jpg",
]

out_dir = "outputs/edge_viz"
os.makedirs(out_dir, exist_ok=True)

def binarize_image(gray):
    # Try Otsu first
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, th_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Also compute adaptive in case of uneven lighting
    th_adapt = cv2.adaptiveThreshold(blur, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 15, 7)
    # Heuristic: choose adaptive if otsu yields too dark/bright result (shadow)
    if np.mean(th_otsu) < 10 or np.mean(th_otsu) > 245:
        return th_adapt
    return th_otsu

for p in image_paths:
    img = cv2.imread(p)
    if img is None:
        print("Failed to read:", p)
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    bin_img = binarize_image(gray)

    # Canny parameters can be tuned per image; these are good defaults
    low, high = 50, 150
    edges = cv2.Canny(bin_img, low, high, apertureSize=3)

    # Save outputs
    base = os.path.splitext(os.path.basename(p))[0]
    cv2.imwrite(os.path.join(out_dir, base + "_gray.jpg"), gray)
    cv2.imwrite(os.path.join(out_dir, base + "_binarized.jpg"), bin_img)
    cv2.imwrite(os.path.join(out_dir, base + "_edges.jpg"), edges)

    # Display inline (matplotlib)
    fig, ax = plt.subplots(1,3, figsize=(12,6))
    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Original")
    ax[1].imshow(bin_img, cmap="gray")
    ax[1].set_title("Binarized")
    ax[2].imshow(edges, cmap="gray")
    ax[2].set_title("Canny edges")
    for a in ax:
        a.axis("off")
    plt.suptitle(base)
    plt.tight_layout()
    plt.show()