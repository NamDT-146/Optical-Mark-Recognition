# Methodology & Algorithmic Framework

This document outlines the core architecture and algorithm behind the End-to-End Optical Mark Recognition (OMR) system used in this project. The logic has been perfectly mirrored across both an exploratory **Python** implementation and a high-performance **C++ Core Engine**.

## 1. The Strategy: Geometry First

Early versions of OMR focused on detecting individual circular bubbles using algorithms like the Hough Circle Transform. This method is brittle and fails against skewed, shadowed, distorted, or low-resolution real-world scanned pages.

Instead, this framework implements a **"Grid Strategy."** We rely on accurately extracting large structural bounding boxes in the document (Exam Code block, Student ID block, Question Options block) and mathematically subdividing them to locate internal options, guided by a fixed JSON generation template. 

## 2. Processing Pipeline

The End-to-End pipeline handles arbitrary page rotations, lighting, and camera orientations.

### Step 1: Image Preprocessing
Before any shape can be found, the image must be converted to binary mask highlighting structures:
- **Grayscaling:** Removing color channels.
- **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Enhances local contrast, heavily mitigating shadows or dark patches without wiping out information.
- **Gaussian Blur:** Smoothes out grain noise.
- **Adaptive Thresholding:** Binarizes the preprocessed layer allowing black printed shapes to stand out cleanly against the paper background, regardless of localized lighting.

### Step 2: Page Warping & Anchor Detection
We use four rectangular anchors dynamically located at the extremes of the scanning templates. 
- Using `cv2.findContours`, we isolate roughly-square shapes (`cv2.boundingRect`) passing strict aspect-ratio and size tests.
- If only 3 anchors are found due to clipping, vector/centroid algorithms infer the 4th anchor point.
- The entire page is perspective-warped (`cv2.warpPerspective`) into a perfectly rectangular, flat digital representation scaling back to its original JSON template resolution (e.g., 2100x2970 px).

### Step 3: Block Extraction (The Grid Strategy)
Once the page is warped and thresholded again:
1. **Isolate large blocks:** We filter contours for specific macroscopic bounding boxes (like the Exam Code block with a `3.83` aspect ratio, or Question Columns with a `~2.6` aspect ratio).
2. **Subdivision calculation:** Based on generating parameters derived from `Professional_OMR_*.json`, the width and height of the box are mathematically divided into X columns and Y rows.
3. **Region of Interest (ROI):** We pinpoint the precise exact location (x, y) where a circle bubbled should exist.

### Step 4: Decision Engine & Fill Evaluation
For each derived sub-block (inner circle):
- A digital mask (`cv2.circle`) is drawn tightly over the estimated coordinate.
- We run a bitwise-and comparing the mask against the binary image, returning the count of "filled" black pixels vs total mask pixels. 
- **The Threshold Check:** If the ink density surpasses `fill_threshold` (e.g., > 35%), the option is considered chosen.

### Step 5: The Grading Execution
To process into actual scores safely:
- The parser evaluates an array of 4 ratios (`A, B, C, D`) per line.
- If multiple options pass the threshold, or a "fuzzy" second option is too close in density to the primary, the parser tags the result as **`M`** (Multiple Choices).
- Comparing against the Excel/CSV Answer key, an `M` is graded as a safe structural `FALSE` (0 points).
- Blank answers are also 0 points.
- Correct single choices tally up, and the raw score is dynamically **scaled to 10 points** before being cleanly exported as a final CSV record containing the visual debugging data (`final_results.csv`).