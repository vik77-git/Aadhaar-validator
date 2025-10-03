# app.py
import os
import re
import io
import cv2
import numpy as np
from datetime import datetime
from PIL import Image
import gradio as gr
from ultralytics import YOLO
import pytesseract

# ---------- CONFIG ----------
# On Hugging Face Spaces, tesseract is available at /usr/bin/tesseract after apt install
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

ROOT = os.path.dirname(__file__)
MODEL_PATH = os.path.join(ROOT, "best.pt")  # put your YOLO best.pt in repo root
os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)

# Make sure this matches the classes used during YOLO training
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
    # img is a BGR numpy array
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except Exception:
        # if single-channel already
        gray = img
    # adaptive thresholding to help OCR
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Convert to PIL for pytesseract reliability
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

# ---------- Load model ----------
print("Loading YOLO model from:", MODEL_PATH)
model = None
try:
    model = YOLO(MODEL_PATH)
    print("YOLO model loaded.")
except Exception as e:
    print("Warning: could not load YOLO model:", e)
    model = None

# ---------- Processing ----------
def pil_to_cv2(image: Image.Image):
    image = image.convert("RGB")
    arr = np.array(image)
    # PIL -> RGB, convert to BGR for OpenCV
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return bgr

def process_image(file_obj):
    # file_obj is a bytes-like / file buffer from Gradio
    try:
        image = Image.open(file_obj)
    except Exception as e:
        return {"error": f"Cannot open image: {e}"}
    img = pil_to_cv2(image)
    h, w = img.shape[:2]

    extracted = {"aadhaar_number": "", "dob": "", "name": "", "photo": ""}

    if model is None:
        return {"error": "YOLO model not loaded. Ensure best.pt is in repo root."}

    # run inference
    try:
        results = model(img)  # ultralytics YOLO: pass numpy image
    except Exception as e:
        return {"error": f"Model inference error: {e}"}

    # parse detections
    for r in results:
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            continue
        # boxes.xyxy: tensor Nx4, boxes.cls: tensor N
        try:
            xyxy = boxes.xyxy.cpu().numpy()  # (N,4)
            cls_ids = boxes.cls.cpu().numpy().astype(int)
        except Exception:
            # fallback if data already numpy
            xyxy = np.array(boxes.xyxy)
            cls_ids = np.array(boxes.cls).astype(int)

        for (x1, y1, x2, y2), cls_id in zip(xyxy, cls_ids):
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            # clamp
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = img[y1:y2, x1:x2].copy()
            label = CLASS_MAP.get(int(cls_id), f"class_{cls_id}")

            if label == "aadhaar_number":
                text = tesseract_ocr(crop)
                digits = re.sub(r"[^0-9]", "", text)
                if len(digits) == 12:
                    extracted["aadhaar_number"] = digits
            elif label in ("dob", "name"):
                text = tesseract_ocr(crop)
                text = re.sub(r"[^A-Za-z0-9\-/ ]", " ", text).strip()
                # prefer longer text
                if len(text) > len(extracted.get(label, "")):
                    extracted[label] = text
            elif label == "photo":
                # save the photo crop and return a path to display
                rel_photo = f"photo_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
                out_photo = os.path.join(ROOT, "outputs", rel_photo)
                cv2.imwrite(out_photo, crop)
                extracted["photo"] = out_photo

    # fallback: full OCR on the entire image if missing fields
    full_text = tesseract_ocr(img)
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

    # prepare display image (annotate detections)
    annotated = img.copy()
    # draw boxes & labels
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

    response = {
        "extracted": extracted,
        "validations": validations,
        "annotated_pil": annotated_pil,
    }
    # convert saved photo path to PIL if exists
    if extracted.get("photo"):
        try:
            response["photo_pil"] = Image.open(extracted["photo"])
        except Exception:
            response["photo_pil"] = None

    return response

# ---------- Gradio UI ----------
with gr.Blocks(css=open("style.css").read() if os.path.exists("style.css") else None, title="Aadhaar Card Verification System") as demo:
    gr.HTML("<div style='text-align:center; margin-top:16px;'><h1>Aadhaar Card Verification System</h1><p class='subtitle'>Upload an Aadhaar card image for validation</p></div>")
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="📂 Choose Aadhaar File (jpg, png)", file_types=["image"], type="filepath")
            scan_btn = gr.Button("🔍 Scan Aadhaar", elem_id="scanButton")
            status_text = gr.Markdown("")
        with gr.Column(scale=1):
            annotated_img = gr.Image(label="Detected / Annotated", type="pil")
            photo_img = gr.Image(label="Aadhaar Photo (if detected)", type="pil")
    # results area
    aadhaar_text = gr.Markdown("", elem_id="resultsDiv")

    def on_scan(file):
        if file is None:
            return gr.update(), gr.update(), gr.update(value="**Error:** Please upload an image file.")
        # Process
        out = process_image(file.name if hasattr(file, "name") else file)
        if isinstance(out, dict) and out.get("error"):
            return None, None, f"**Error:** {out.get('error')}"
        # Prepare results markdown
        extracted = out["extracted"]
        validations = out["validations"]
        aadhaar_num = extracted.get("aadhaar_number") or "❌ Missing"
        name = extracted.get("name") or "❌ Missing"
        dob = extracted.get("dob") or "❌ Missing"
        results_md = f"""
### Extracted Data
- **Aadhaar Number:** `{aadhaar_num}`  — **{validations['aadhaar_number']}**
- **Name:** `{name}`  — **{validations['name']}**
- **DOB:** `{dob}`  — **{validations['dob']}**
"""
        return out.get("annotated_pil"), out.get("photo_pil"), results_md

    scan_btn.click(on_scan, inputs=[file_input], outputs=[annotated_img, photo_img, aadhaar_text])

    gr.HTML("<footer style='text-align:center; margin-top:18px; padding:10px; background:#1a73e8; color:#fff'>© 2025 Aadhaar Verification System | Secure Identity Validation</footer>")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)