import os
import re
import io
import sys
import uuid
import shutil
import cv2
import numpy as np
from datetime import datetime
from PIL import Image
from flask import Flask, request, render_template, url_for, redirect
from ultralytics import YOLO
import pytesseract

# ---------- CONFIG ----------
# Prefer explicit tesseract path if present (Spaces installs via packages.txt)
if shutil.which("tesseract"):
    pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract")
else:
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

ROOT = os.path.dirname(__file__)
MODEL_PATH = os.path.join(ROOT, "best.pt")
UPLOAD_FOLDER = os.path.join(ROOT, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Make sure this matches your model's class ordering
CLASS_MAP = {0: "aadhaar_card", 1: "aadhaar_number", 2: "dob", 3: "name", 4: "photo"}

# ---------- Verhoeff (Aadhaar) ----------
verhoeff_table_d = [
    [0,1,2,3,4,5,6,7,8,9],
    [1,2,3,4,0,6,7,8,9,5],
    [2,3,4,0,1,7,8,9,5,6],
    [3,4,0,1,2,8,9,5,6,7],
    [4,0,1,2,3,9,5,6,7,8],
    [5,9,8,7,6,0,4,3,2,1],
    [6,5,9,8,7,1,0,4,3,2],
    [7,6,5,9,8,2,1,0,4,3],
    [8,7,6,5,9,3,2,1,0,4],
    [9,8,7,6,5,4,3,2,1,0]
]
verhoeff_table_p = [
    [0,1,2,3,4,5,6,7,8,9],
    [1,5,7,6,2,8,3,0,9,4],
    [5,8,0,3,7,9,6,1,4,2],
    [8,9,1,6,0,4,3,5,2,7],
    [9,4,5,3,1,2,6,8,7,0],
    [4,2,8,6,5,7,3,9,0,1],
    [2,7,9,3,8,0,6,4,1,5],
    [7,0,4,6,9,1,3,2,5,8]
]
def verhoeff_check(num):
    c = 0
    num = num[::-1]
    for i, item in enumerate(num):
        c = verhoeff_table_d[c][verhoeff_table_p[i % 8][int(item)]]
    return c == 0

# ---------- Utilities ----------
ALLOWED_EXT = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def tesseract_ocr(img):
    """
    img: BGR numpy array
    returns recognized text (string) or raises RuntimeError if tesseract missing
    """
    # ensure tesseract binary available
    if not shutil.which(pytesseract.pytesseract.tesseract_cmd) and not shutil.which("tesseract"):
        raise RuntimeError("Tesseract binary not found on the system. Install tesseract-ocr (packages.txt for Spaces).")
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except Exception:
        gray = img
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    pil_img = Image.fromarray(th)
    text = pytesseract.image_to_string(pil_img, config="--psm 6")
    return text.strip()

def validate_name(name):
    if not name: return "❌ Missing"
    cleaned = re.sub(r"[^A-Za-z\s\.]", "", name).strip()
    return "✅ Valid" if len(cleaned) >= 3 else "❌ Invalid"

def validate_dob(dob_str):
    if not dob_str: return "❌ Missing"
    patterns = ["%d-%m-%Y", "%d/%m/%Y", "%Y"]
    for p in patterns:
        try:
            s = re.sub(r"[^0-9\-\/]", "", dob_str)
            dt = datetime.strptime(s, p)
            today = datetime.today()
            if dt > today: return "❌ Future DOB"
            age = today.year - dt.year - ((today.month, today.day) < (dt.month, dt.day))
            if age < 5: return "❌ Age too low (<5)"
            return "✅ Valid"
        except:
            continue
    return "❌ Invalid format"

def validate_aadhaar(num):
    if not num: return "❌ Missing"
    digits = re.sub(r"[^0-9]", "", num)
    if len(digits) != 12: return "❌ Invalid length"
    return "✅ Valid" if verhoeff_check(digits) else "❌ Invalid (Verhoeff failed)"

# ---------- Load YOLO model ----------
print("Loading YOLO model from:", MODEL_PATH)
model = None
try:
    model = YOLO(MODEL_PATH)
    print("YOLO model loaded.")
except Exception as e:
    print("Warning: YOLO model not loaded:", e, file=sys.stderr)
    model = None

# ---------- Helpers to open images ----------
def pil_to_cv2(image: Image.Image):
    image = image.convert("RGB")
    arr = np.array(image)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return bgr

def open_input_image(input_obj):
    """
    Accepts:
      - filepath (str)
      - file-like (has .read())
      - bytes
    Returns PIL.Image
    """
    if input_obj is None:
        raise ValueError("No input provided")
    if isinstance(input_obj, str):
        return Image.open(input_obj)
    if hasattr(input_obj, "read"):
        data = input_obj.read()
        if isinstance(data, bytes):
            return Image.open(io.BytesIO(data))
        return Image.open(io.BytesIO(data))
    if isinstance(input_obj, (bytes, bytearray)):
        return Image.open(io.BytesIO(input_obj))
    raise ValueError(f"Unsupported image input type: {type(input_obj)}")

# ---------- Core processing ----------
def process_image_file(file_stream_or_path):
    """Returns dict: extracted, validations, annotated_path, photo_path (if any)"""
    try:
        pil_image = open_input_image(file_stream_or_path)
    except Exception as e:
        return {"error": f"Cannot open image: {e}"}

    img = pil_to_cv2(pil_image)
    h, w = img.shape[:2]

    extracted = {"aadhaar_number": "", "dob": "", "name": "", "photo": ""}

    if model is None:
        return {"error": "YOLO model not loaded. Ensure best.pt is present in repo root."}

    try:
        results = model(img)
    except Exception as e:
        return {"error": f"Model inference error: {e}"}

    # parse detections
    for r in results:
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            continue
        try:
            xyxy = boxes.xyxy.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)
        except Exception:
            xyxy = np.array(boxes.xyxy)
            cls_ids = np.array(boxes.cls).astype(int)

        for (x1, y1, x2, y2), cls_id in zip(xyxy, cls_ids):
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = img[y1:y2, x1:x2].copy()
            label = CLASS_MAP.get(int(cls_id), f"class_{cls_id}")

            if label == "aadhaar_number":
                try:
                    text = tesseract_ocr(crop)
                except Exception as e:
                    return {"error": f"Tesseract OCR error: {e}"}
                digits = re.sub(r"[^0-9]", "", text)
                if len(digits) == 12:
                    extracted["aadhaar_number"] = digits
            elif label in ("dob", "name"):
                try:
                    text = tesseract_ocr(crop)
                except Exception as e:
                    return {"error": f"Tesseract OCR error: {e}"}
                text = re.sub(r"[^A-Za-z0-9\-/ ]", " ", text).strip()
                if len(text) > len(extracted.get(label, "")):
                    extracted[label] = text
            elif label == "photo":
                rel_photo = f"photo_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
                out_photo = os.path.join(UPLOAD_FOLDER, rel_photo)
                cv2.imwrite(out_photo, crop)
                extracted["photo"] = out_photo

    # fallback full OCR
    try:
        full_text = tesseract_ocr(img)
    except Exception as e:
        return {"error": f"Tesseract OCR error: {e}"}

    if not extracted["aadhaar_number"]:
        m = re.search(r"\b\d{4}\s?\d{4}\s?\d{4}\b", full_text)
        if m:
            extracted["aadhaar_number"] = re.sub(r"\s+", "", m.group(0))
    lines = [ln.strip() for ln in full_text.splitlines() if ln.strip()]
    if not extracted["dob"]:
        for ln in lines:
            m = re.search(r'(DOB|DoB|D\.O\.B|YOB|Year of Birth)[:\s]*([0-9]{2}[-/][0-9]{2}[-/][0-9]{4}|\d{4})', ln, re.IGNORECASE)
            if m:
                extracted["dob"] = m.group(2)
                break
    if not extracted["name"] and extracted["dob"]:
        for i, ln in enumerate(lines):
            if extracted["dob"] in ln and i > 0:
                candidate = re.sub(r"[^A-Za-z\s\.]", "", lines[i - 1]).strip()
                if len(candidate) >= 3:
                    extracted["name"] = candidate
                break

    validations = {
        "aadhaar_number": validate_aadhaar(extracted.get("aadhaar_number")),
        "name": validate_name(extracted.get("name")),
        "dob": validate_dob(extracted.get("dob")),
    }

    # annotated image
    annotated = img.copy()
    try:
        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), cls_id in zip(xyxy, cls_ids):
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                color = (50, 150, 255)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, CLASS_MAP.get(int(cls_id), str(cls_id)), (x1, max(0, y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    except Exception:
        pass

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    annotated_pil = Image.fromarray(annotated_rgb)

    # save annotated image
    ann_name = f"annotated_{uuid.uuid4().hex}.jpg"
    ann_path = os.path.join(UPLOAD_FOLDER, ann_name)
    annotated_bgr = cv2.cvtColor(np.array(annotated_pil), cv2.COLOR_RGB2BGR)
    cv2.imwrite(ann_path, annotated_bgr)

    response = {
        "extracted": extracted,
        "validations": validations,
        "annotated_path": ann_path,
        "photo_path": extracted.get("photo")
    }

    return response

# ---------- Flask app ----------
from flask import Flask, render_template
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods=["GET"])
def index_route():
    return render_template("index.html")

@app.route("/scan", methods=["POST"])
def scan_route():
    if "file" not in request.files:
        return render_template("index.html", error="No file part in request")
    file = request.files["file"]
    if file.filename == "":
        return render_template("index.html", error="No file selected")
    if not allowed_file(file.filename):
        return render_template("index.html", error="Invalid file type")

    # save uploaded file
    fname = f"upload_{uuid.uuid4().hex}.jpg"
    save_path = os.path.join(UPLOAD_FOLDER, fname)
    file.save(save_path)

    # process
    result = process_image_file(save_path)
    if isinstance(result, dict) and result.get("error"):
        return render_template("index.html", error=result.get("error"))

    extracted = result["extracted"]
    validations = result["validations"]
    annotated_rel = os.path.relpath(result["annotated_path"], ROOT)
    photo_rel = os.path.relpath(result.get("photo_path") or "", ROOT)

    return render_template(
        "result.html",
        input_image=url_for('static', filename=f"uploads/{fname}"),
        annotated_image=url_for('static', filename=f"{os.path.basename(annotated_rel)}"),
        photo_image=(url_for('static', filename=f"{os.path.basename(photo_rel)}") if photo_rel else None),
        extracted=extracted,
        validations=validations
    )

if __name__ == "__main__":
    # Port 7860 for Hugging Face Spaces
    app.run(host="0.0.0.0", port=7860)