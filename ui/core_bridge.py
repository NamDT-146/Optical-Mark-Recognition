import sys
import os
import cv2
import json

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from experiments.pdf_generate.dynamic_advance import ProfessionalGenerator
from experiments.images_parser.robust_scanner import RobustOMRScanner
from experiments.answer_parser.answer_parser import AnswerParser

def generate_template(num_qs=45, output_pdf="ui/static/template.pdf"):
    gen = ProfessionalGenerator(output_pdf)
    gen.generate(num_qs, save_pdf=True)
    return output_pdf, output_pdf.replace(".pdf", ".json")

def process_image(image_path, output_path="ui/static/warped.png"):
    scanner = RobustOMRScanner()
    img = cv2.imread(image_path)
    if img is None:
        return None, "Invalid image"
    detect_res = scanner.detect_anchors(img)
    if not detect_res:
        return None, "Failed to detect anchors"
    anchors, _ = detect_res
    warped, _ = scanner.warp_perspective(img, anchors)
    cv2.imwrite(output_path, warped)
    return output_path, "Success"

def parse_answers(warped_img_path, template_json_path, output_debug_path="ui/static/debug.png"):
    parser = AnswerParser(template_json_path)
    img = cv2.imread(warped_img_path)
    if img is None:
        return None
    answers = parser.parse_answers(img, debug_output_path=output_debug_path)
    return answers
