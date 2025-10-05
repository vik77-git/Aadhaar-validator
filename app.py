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
from flask import Flask, request, render_template, url_for, redirect, send_from_directory
from ultralytics import YOLO
import pytesseract
from supabase import create_client, Client

# ---------- CONFIG ----------
if shutil.which("tesseract"):
    pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract")
else:
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

ROOT = os.path.dirname(__file__)
MODEL_PATH = os.path.join(ROOT, "best.pt")

UPLOAD_FOLDER = os.path.join("/tmp", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

CLASS_MAP = {0: "aadhaar_card", 1: "aadhaar_number", 2: "dob", 3: "name", 4: "photo"}

# ---------- SUPABASE SETUP ----------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "aadhaar-photos")

supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Supabase client initialized.")
    except Exception as e:
        supabase = None
        print("⚠️ Supabase init error:", e, file=sys.stderr)
else:
    print("⚠️ Missing Supabase credentials in environment. Database/uploads will be skipped.", file=sys.stderr)

# ---------- VERHOEFF ----------
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

# ---------- UTILITIES ----------
ALLOWED_EXT = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def tesseract_ocr(img):
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

# ---------- LOAD YOLO ----------
print("Loading YOLO model from:", MODEL_PATH)
try:
    model = YOLO(MODEL_PATH)
    print("✅ YOLO model loaded.")
except Exception as e:
    model = None
    print("⚠️ YOLO model not loaded:", e)

# ---------- IMAGE HELPERS ----------
def pil_to_cv2(image: Image.Image):
    image = image.convert("RGB")
    arr = np.array(image)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def open_input_image(input_obj):
    if input_obj is None:
        raise ValueError("No input provided")
    if isinstance(input_obj, str):
        return Image.open(input_obj)
    if hasattr(input_obj, "read"):
        data = input_obj.read()
        return Image.open(io.BytesIO(data))
    if isinstance(input_obj, (bytes, bytearray)):
        return Image.open(io.BytesIO(input_obj))
    raise ValueError(f"Unsupported image input type: {type(input_obj)}")

# ---------- SUPABASE UPLOAD ----------
def upload_to_supabase(local_path, orig_filename):
    if not supabase:
        print("⚠️ Supabase not initialized -- skipping upload.")
        return None

    unique_name = f"{uuid.uuid4().hex}_{os.path.basename(orig_filename)}"
    try:
        with open(local_path, "rb") as f:
            res = supabase.storage.from_(SUPABASE_BUCKET).upload(unique_name, f, {"upsert": True})
        pub = supabase.storage.from_(SUPABASE_BUCKET).get_public_url(unique_name)
        url = pub if isinstance(pub, str) else pub.get("publicURL") or pub.get("publicUrl")
        print("✅ Uploaded file public URL:", url)
        return url
    except Exception as e:
        print("⚠️ Error uploading to Supabase:", e, file=sys.stderr)
        return None

# ---------- MAIN PROCESS ----------
def process_image_file(file_stream_or_path):
    try:
        pil_image = open_input_image(file_stream_or_path)
    except Exception as e:
        return {"error": f"Cannot open image: {e}"}

    img = pil_to_cv2(pil_image)
    h, w = img.shape[:2]
    extracted = {"aadhaar_number": "", "dob": "", "name": "", "photo": ""}

    if model is None:
        return {"error": "YOLO model not loaded."}

    try:
        results = model(img)
    except Exception as e:
        return {"error": f"Model inference error: {e}"}

    for r in results:
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), cls_id in zip(xyxy, cls_ids):
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
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
                if len(text) > len(extracted.get(label, "")):
                    extracted[label] = text
            elif label == "photo":
                rel_photo = f"photo_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
                out_photo = os.path.join(UPLOAD_FOLDER, rel_photo)
                cv2.imwrite(out_photo, crop)
                extracted["photo"] = rel_photo

    # OCR fallback
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

    # Annotated result
    ann_name = f"annotated_{uuid.uuid4().hex}.jpg"
    ann_path = os.path.join(UPLOAD_FOLDER, ann_name)
    annotated = img.copy()
    for r in results:
        for (x1, y1, x2, y2), cls_id in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)):
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (50, 150, 255), 2)
            cv2.putText(annotated, CLASS_MAP.get(cls_id, str(cls_id)), (int(x1), max(0, int(y1)-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 150, 255), 2)
    cv2.imwrite(ann_path, annotated)

    # Upload photo to Supabase
    photo_url = None
    if extracted.get("photo"):
        local_photo = os.path.join(UPLOAD_FOLDER, extracted["photo"])
        print("Attempting to upload photo:", local_photo)
        photo_url = upload_to_supabase(local_photo, extracted["photo"])

    # Insert into Supabase
    try:
        if supabase:
            validity_status = "Valid" if all(v.startswith("✅") for v in validations.values()) else "Invalid"
            record = {
                "id": str(uuid.uuid4()),
                "name": extracted.get("name"),
                "aadhaar_no": extracted.get("aadhaar_number"),
                "dob": extracted.get("dob"),
                "photo_url": photo_url or "",
                "validity": validity_status,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                "verified_at": datetime.utcnow().isoformat()
            }
            print("📄 Extracted data before insert:", extracted)
            print("🧾 Inserting record to Supabase:", record)
            resp = supabase.table("verifications").insert([record]).execute()
            print("✅ Supabase insert response:", repr(resp))
    except Exception as e:
        print("⚠️ Supabase DB insert failed:", e, file=sys.stderr)

    return {
        "extracted": extracted,
        "validations": validations,
        "annotated": os.path.basename(ann_path),
        "photo": extracted.get("photo")
    }

# ---------- FLASK ----------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/instructions')
def instructions():
    return render_template('instructions.html')

@app.route("/scan", methods=["POST"])
def scan_route():
    if "file" not in request.files:
        return render_template("index.html", error="No file part in request")
    file = request.files["file"]
    if file.filename == "":
        return render_template("index.html", error="No file selected")
    if not allowed_file(file.filename):
        return render_template("index.html", error="Invalid file type")

    fname = f"upload_{uuid.uuid4().hex}.jpg"
    save_path = os.path.join(UPLOAD_FOLDER, fname)
    file.save(save_path)

    result = process_image_file(save_path)
    if result.get("error"):
        return render_template("index.html", error=result["error"])

    return render_template(
        "result.html",
        input_image=url_for("serve_upload", filename=fname),
        annotated_image=url_for("serve_upload", filename=result["annotated"]),
        photo_image=(url_for("serve_upload", filename=result["photo"]) if result.get("photo") else None),
        extracted=result["extracted"],
        validations=result["validations"]
    )

@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)