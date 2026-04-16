import cv2
import json
import os
from pathlib import Path
from page_scanner import PageBasedOMRScanner

class PageBasedOMRParser:
    """
    Page-based OMR parser: Detects paper boundary, then extracts ROIs.
    Much more robust than anchor-based approach.
    """
    
    def __init__(self, json_path, min_paper_area=50000):
        with open(json_path, "r") as f:
            self.config = json.load(f)
        self.scanner = PageBasedOMRScanner(min_paper_area=min_paper_area)

    def extract_from_warped(self, warped_img, page_idx=0):
        """Extract ROIs from warped image based on JSON coordinates"""
        page_config = self.config["pages"][page_idx]
        extracted_data = {}

        for roi_name, coords in page_config["rois"].items():
            # Handle both standard (w, h) and questions-format (col_w, row_h)
            x = coords.get("x")
            y = coords.get("y")
            w = coords.get("w") or coords.get("col_w")
            h = coords.get("h") or coords.get("row_h")
            
            # Skip if required coordinates missing
            if not all([x, y, w, h]):
                continue
            
            # Crop ROI
            roi_img = warped_img[y : y+h, x : x+w]
            extracted_data[roi_name] = roi_img
            
            # Debug: Draw ROI rectangles
            cv2.rectangle(warped_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
        return extracted_data, warped_img
    
    def detect_and_warp(self, image):
        """Detect page and warp to standard size"""
        result = self.scanner.detect_page(image)
        
        if not result:
            return None, False
        
        corners, _ = result
        warped, _ = self.scanner.warp_perspective(image, corners)
        
        return warped, True


def test_page_scanner_parser(config_json, images_dir="test/images", output_dir="outputs"):
    """Full pipeline: Page detection → Warp → ROI Extraction"""
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/page_debug", exist_ok=True)
    os.makedirs(f"{output_dir}/roi_debug", exist_ok=True)
    os.makedirs(f"{output_dir}/extracted_rois", exist_ok=True)
    os.makedirs(f"{output_dir}/warped", exist_ok=True)
    
    parser = PageBasedOMRParser(config_json)
    results = []
    
    # Get all images
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = sorted([
        f for f in os.listdir(images_dir)
        if Path(f).suffix.lower() in image_extensions
    ])
    
    if not image_files:
        print(f"No images found in {images_dir}")
        return
    
    print(f"Found {len(image_files)} images\n")
    
    for idx, img_file in enumerate(image_files):
        img_path = os.path.join(images_dir, img_file)
        print(f"[{idx+1}/{len(image_files)}] {img_file}")
        
        try:
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                print(f"  ❌ Failed to load")
                continue
            
            print(f"  ↳ Original: {img.shape[1]}×{img.shape[0]}")
            
            # Step 1: Detect page and warp
            result = parser.scanner.detect_page(img)
            if not result:
                print(f"  ❌ Page detection failed")
                results.append({"image": img_file, "error": "Page detection failed"})
                continue
            
            corners, debug_img = result
            
            # Save page detection debug
            page_debug_path = f"{output_dir}/page_debug/{Path(img_file).stem}_page.png"
            cv2.imwrite(page_debug_path, debug_img)
            print(f"  ✓ Page detected")
            
            # Step 2: Warp
            warped, M = parser.scanner.warp_perspective(img, corners)
            warped_path = f"{output_dir}/warped/{Path(img_file).stem}_warped.png"
            cv2.imwrite(warped_path, warped)
            print(f"  ✓ Warped: {warped.shape[1]}×{warped.shape[0]}")
            
            # Step 3: Extract ROIs
            extracted_data, roi_debug_img = parser.extract_from_warped(warped, page_idx=0)
            
            # Save ROI debug
            roi_debug_path = f"{output_dir}/roi_debug/{Path(img_file).stem}_rois.png"
            cv2.imwrite(roi_debug_path, roi_debug_img)
            
            # Save individual ROIs
            roi_output_dir = f"{output_dir}/extracted_rois/{Path(img_file).stem}"
            os.makedirs(roi_output_dir, exist_ok=True)
            
            for roi_name, roi_img in extracted_data.items():
                roi_path = f"{roi_output_dir}/{roi_name}.png"
                cv2.imwrite(roi_path, roi_img)
            
            # Record result
            result = {
                "image": img_file,
                "original_size": [img.shape[1], img.shape[0]],
                "warped_size": [warped.shape[1], warped.shape[0]],
                "corners": corners,
                "rois_extracted": list(extracted_data.keys()),
                "page_debug": page_debug_path,
                "warped": warped_path,
                "roi_debug": roi_debug_path,
                "roi_output_dir": roi_output_dir
            }
            results.append(result)
            print(f"  ✓ Success! {len(extracted_data)} ROIs extracted\n")
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)}\n")
            results.append({"image": img_file, "error": str(e)})
    
    # Save summary
    results_path = f"{output_dir}/page_scanner_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    
    # Print summary
    success_count = sum(1 for r in results if "error" not in r)
    print(f"\n{'='*60}")
    print(f"Results: {success_count}/{len(image_files)} images processed")
    print(f"{'='*60}")
    print(f"✓ Page detection debug: {output_dir}/page_debug/")
    print(f"✓ ROI extraction debug: {output_dir}/roi_debug/")
    print(f"✓ Warped images: {output_dir}/warped/")
    print(f"✓ Extracted ROIs: {output_dir}/extracted_rois/")
    print(f"✓ Summary: {results_path}")


if __name__ == "__main__":
    import sys
    
    # Use absolute paths
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_file = os.path.join(root_dir, "test/templates/Professional_OMR_45.json")
    images_dir = os.path.join(root_dir, "test/images")
    output_dir = os.path.join(root_dir, "outputs/PageScanner_OMR_Test")
    
    print("=== Page-Based OMR Scanner with ROI Extraction ===\n")
    
    test_page_scanner_parser(
        config_json=config_file,
        images_dir=images_dir,
        output_dir=output_dir
    )
