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
from flask import Flask, request, render_template, url_for, send_from_directory
from ultralytics import YOLO
import pytesseract

# ---------- Supabase ----------
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

# ---------- Supabase Setup ----------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "aadhaar-photos")

supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Supabase connected.")
    except Exception as e:
        print("⚠️ Supabase init error:", e)
else:
    print("⚠️ Missing Supabase credentials.")

# ---------- Verhoeff (Aadhaar Validation) ----------
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
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(Image.fromarray(th), config="--psm 6")
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
            if dt > datetime.today(): return "❌ Future DOB"
            age = datetime.today().year - dt.year
            if age < 5: return "❌ Age too low (<5)"
            return "✅ Valid"
        except:
            continue
    return "❌ Invalid"

def validate_aadhaar(num):
    digits = re.sub(r"[^0-9]", "", num)
    if len(digits) != 12: return "❌ Invalid length"
    return "✅ Valid" if verhoeff_check(digits) else "❌ Invalid"

# ---------- YOLO ----------
print("Loading YOLO model...")
try:
    model = YOLO(MODEL_PATH)
    print("✅ YOLO loaded.")
except Exception as e:
    print("⚠️ YOLO load failed:", e)
    model = None

# ---------- Supabase Upload ----------
def upload_to_supabase(local_path, filename):
    if not supabase:
        print("⚠️ Supabase not initialized.")
        return None
    try:
        with open(local_path, "rb") as f:
            supabase.storage.from_(SUPABASE_BUCKET).upload(filename, f)
        return supabase.storage.from_(SUPABASE_BUCKET).get_public_url(filename)
    except Exception as e:
        print("⚠️ Upload error:", e)
        return None

# ---------- Core Processing ----------
def process_image_file(path):
    img = cv2.imread(path)
    h, w = img.shape[:2]
    extracted = {"aadhaar_number": "", "dob": "", "name": "", "photo": ""}
    if not model:
        return {"error": "Model not loaded"}

    results = model(img)
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), cid in zip(boxes, classes):
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            label = CLASS_MAP.get(cid, "unknown")
            crop = img[y1:y2, x1:x2]
            if label in ("aadhaar_number", "dob", "name"):
                text = tesseract_ocr(crop)
                if label == "aadhaar_number":
                    digits = re.sub(r"[^0-9]", "", text)
                    if len(digits) == 12: extracted["aadhaar_number"] = digits
                else:
                    extracted[label] = text.strip()
            elif label == "photo":
                pname = f"photo_{uuid.uuid4().hex}.jpg"
                outp = os.path.join(UPLOAD_FOLDER, pname)
                cv2.imwrite(outp, crop)
                extracted["photo"] = pname

    full_text = tesseract_ocr(img)
    if not extracted["aadhaar_number"]:
        m = re.search(r"\b\d{4}\s?\d{4}\s?\d{4}\b", full_text)
        if m:
            extracted["aadhaar_number"] = re.sub(r"\s+", "", m.group())

    validations = {
        "aadhaar_number": validate_aadhaar(extracted["aadhaar_number"]),
        "name": validate_name(extracted["name"]),
        "dob": validate_dob(extracted["dob"]),
    }

    ann_name = f"annotated_{uuid.uuid4().hex}.jpg"
    ann_path = os.path.join(UPLOAD_FOLDER, ann_name)
    annotated = img.copy()
    for r in results:
        for (x1, y1, x2, y2), cid in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)):
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(annotated, CLASS_MAP.get(cid, str(cid)), (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.imwrite(ann_path, annotated)

    # ✅ Upload photo + insert record
    photo_url = None
    if extracted["photo"]:
        local_photo = os.path.join(UPLOAD_FOLDER, extracted["photo"])
        photo_url = upload_to_supabase(local_photo, extracted["photo"])

    validity = "Valid" if all(v == "✅ Valid" for v in validations.values()) else "Invalid"

    try:
        if supabase:
            supabase.table("verifications").insert({
                "id": str(uuid.uuid4()),
                "aadhaar_number": extracted["aadhaar_number"],
                "name": extracted["name"],
                "dob": extracted["dob"],
                "photo_url": photo_url or "",
                "validity": validity,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                "verified_at": datetime.utcnow().isoformat()
            }).execute()
            print("✅ Record inserted in Supabase.")
    except Exception as e:
        print("⚠️ Supabase insert failed:", e)

    return {
        "extracted": extracted,
        "validations": validations,
        "annotated": os.path.basename(ann_path),
        "photo": extracted.get("photo")
    }

# ---------- Flask ----------
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
        return render_template("index.html", error="No file uploaded.")
    file = request.files["file"]
    if file.filename == "":
        return render_template("index.html", error="No file selected.")
    if not allowed_file(file.filename):
        return render_template("index.html", error="Invalid file type.")

    fname = f"upload_{uuid.uuid4().hex}.jpg"
    path = os.path.join(UPLOAD_FOLDER, fname)
    file.save(path)

    result = process_image_file(path)
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
