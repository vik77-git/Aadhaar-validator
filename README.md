# Aadhaar Validator (Flask + YOLO + Tesseract)

Aadhaar Card Verification System using Flask, YOLOv8, and Tesseract OCR.

## Included files
- `app.py` — Flask server (YOLO inference, OCR, Verhoeff validation)
- `requirements.txt` — Python dependencies
- `packages.txt` — system packages (Tesseract) for Hugging Face Spaces
- `templates/index.html` — upload UI
- `templates/result.html` — results UI
- `static/style.css` — professional styling
- `static/uploads/` — created at runtime for images
- `best.pt` — **place your YOLO model here (required)**

## Deploy on Hugging Face Spaces
1. Create a new Space (choose **"Other"** / default settings).
2. Upload all files above and **put `best.pt` in the repo root**.
3. The Space will install Python packages from `requirements.txt` and apt packages from `packages.txt`.
4. The Flask app listens on port **7860**; Spaces will expose it automatically.

## Run locally
```bash
pip install -r requirements.txt
# On Ubuntu:
sudo apt-get update && sudo apt-get install -y tesseract-ocr libtesseract-dev
python app.py
