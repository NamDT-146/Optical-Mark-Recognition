from __future__ import annotations

import time
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from core_bridge import generate_template, parse_answers, process_image

app = Flask(__name__, static_folder="static", template_folder="templates")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"
OUTPUT_DIR = STATIC_DIR / "processed"
GENERATED_DIR = STATIC_DIR / "generated"

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

for folder in (UPLOAD_DIR, OUTPUT_DIR, GENERATED_DIR):
	folder.mkdir(parents=True, exist_ok=True)


def _static_url_to_path(url: str | None) -> Path | None:
	if not url or not url.startswith("/static/"):
		return None
	rel_path = url[len("/static/") :]
	path = (STATIC_DIR / rel_path).resolve()
	if STATIC_DIR not in path.parents:
		return None
	return path


@app.route("/")
def index() -> str:
	return render_template("index.html")


@app.route("/api/generate-template", methods=["POST"])
def api_generate_template():
	payload = request.get_json(silent=True) or {}
	num_qs = int(payload.get("num_qs", 45))
	num_qs = max(1, min(300, num_qs))

	stamp = int(time.time() * 1000)
	pdf_name = f"template_{num_qs}_{stamp}.pdf"
	pdf_path = GENERATED_DIR / pdf_name

	generated_pdf, generated_json = generate_template(num_qs=num_qs, output_pdf=str(pdf_path))

	return jsonify(
		{
			"pdf_url": f"/static/generated/{Path(generated_pdf).name}",
			"json_url": f"/static/generated/{Path(generated_json).name}",
			"num_qs": num_qs,
		}
	)


@app.route("/api/process-image", methods=["POST"])
def api_process_image():
	if "image" not in request.files:
		return jsonify({"error": "Missing image file"}), 400

	file = request.files["image"]
	if not file or file.filename == "":
		return jsonify({"error": "No file selected"}), 400

	ext = Path(secure_filename(file.filename)).suffix.lower()
	if ext not in ALLOWED_EXTENSIONS:
		return jsonify({"error": "Unsupported file type"}), 400

	stamp = int(time.time() * 1000)
	src_name = f"upload_{stamp}{ext}"
	src_path = UPLOAD_DIR / src_name
	file.save(src_path)

	warped_name = f"warped_{stamp}.png"
	warped_path = OUTPUT_DIR / warped_name
	result_path, message = process_image(str(src_path), output_path=str(warped_path))

	if not result_path:
		return jsonify({"error": message}), 400

	return jsonify(
		{
			"message": message,
			"original_url": f"/static/uploads/{src_name}",
			"warped_url": f"/static/processed/{warped_name}",
		}
	)


@app.route("/api/parse-answers", methods=["POST"])
def api_parse_answers():
	payload = request.get_json(silent=True) or {}
	warped_url = payload.get("warped_url")
	template_json_url = payload.get("template_json_url")

	warped_path = _static_url_to_path(warped_url)
	template_json_path = _static_url_to_path(template_json_url)

	if not warped_path or not template_json_path:
		return jsonify({"error": "Invalid file selection"}), 400

	if not warped_path.exists() or not template_json_path.exists():
		return jsonify({"error": "File not found"}), 404

	stamp = int(time.time() * 1000)
	debug_name = f"answers_debug_{stamp}.png"
	debug_path = OUTPUT_DIR / debug_name

	answers = parse_answers(
		str(warped_path),
		str(template_json_path),
		output_debug_path=str(debug_path),
	)

	if answers is None:
		return jsonify({"error": "Failed to parse answers"}), 400

	return jsonify(
		{
			"exam_code": answers.get("exam_code", ""),
			"student_id": answers.get("student_id", ""),
			"questions": answers.get("questions", {}),
			"debug_url": f"/static/processed/{debug_name}",
		}
	)


if __name__ == "__main__":
	app.run(host="127.0.0.1", port=5000, debug=True)
