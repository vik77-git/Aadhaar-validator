# Aadhaar Validator - Hugging Face Space (Gradio)

Place `best.pt` (YOLO model file) in the repository root. This Space runs a Gradio app which:

- Loads your YOLO model (`best.pt`) using `ultralytics`.
- Accepts Aadhaar card image uploads and runs detection for: aadhaar_card, aadhaar_number, dob, name, photo.
- Runs Tesseract OCR on detected crops and full image fallback.
- Validates Aadhaar using Verhoeff algorithm.
- Displays annotated image and extracted fields.

Files:
- `app.py` - Gradio application and processing logic
- `style.css` - small UI styles
- `requirements.txt` - Python packages
- `apt.txt` - system packages (Tesseract)
- `best.pt` - put your trained YOLO weights here

Upload all files to the Space and start. If model fails to load, check `best.pt` path and that it's a YOLOv8-compatible weights file.
