import os, re, io, sys, uuid, shutil, cv2, numpy as np
from datetime import datetime
from PIL import Image
from flask import Flask, request, render_template, url_for
from ultralytics import YOLO
import pytesseract

# ---------- CONFIG ----------
if shutil.which("tesseract"):
    pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract")
else:
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

ROOT = os.path.dirname(__file__)
MODEL_PATH = os.path.join(ROOT, "best.pt")
UPLOAD_FOLDER = os.path.join(ROOT, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
CLASS_MAP = {0:"aadhaar_card",1:"aadhaar_number",2:"dob",3:"name",4:"photo"}

# ---------- Verhoeff ----------
verhoeff_table_d=[[0,1,2,3,4,5,6,7,8,9],[1,2,3,4,0,6,7,8,9,5],[2,3,4,0,1,7,8,9,5,6],[3,4,0,1,2,8,9,5,6,7],[4,0,1,2,3,9,5,6,7,8],[5,9,8,7,6,0,4,3,2,1],[6,5,9,8,7,1,0,4,3,2],[7,6,5,9,8,2,1,0,4,3],[8,7,6,5,9,3,2,1,0,4],[9,8,7,6,5,4,3,2,1,0]]
verhoeff_table_p=[[0,1,2,3,4,5,6,7,8,9],[1,5,7,6,2,8,3,0,9,4],[5,8,0,3,7,9,6,1,4,2],[8,9,1,6,0,4,3,5,2,7],[9,4,5,3,1,2,6,8,7,0],[4,2,8,6,5,7,3,9,0,1],[2,7,9,3,8,0,6,4,1,5],[7,0,4,6,9,1,3,2,5,8]]
def verhoeff_check(num):
    c=0
    num=num[::-1]
    for i,item in enumerate(num):
        c=verhoeff_table_d[c][verhoeff_table_p[i%8][int(item)]]
    return c==0

# ---------- Utilities ----------
ALLOWED_EXT={"png","jpg","jpeg"}
def allowed_file(filename):
    return "." in filename and filename.rsplit(".",1)[1].lower() in ALLOWED_EXT

def tesseract_ocr(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
    _,th=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return pytesseract.image_to_string(Image.fromarray(th),config="--psm 6").strip()

def validate_name(name):
    if not name: return "❌ Missing"
    cleaned=re.sub(r"[^A-Za-z\s\.]","",name).strip()
    return "✅ Valid" if len(cleaned)>=3 else "❌ Invalid"

def validate_dob(dob_str):
    if not dob_str: return "❌ Missing"
    patterns=["%d-%m-%Y","%d/%m/%Y","%Y"]
    for p in patterns:
        try:
            s=re.sub(r"[^0-9\-\/]","",dob_str)
            dt=datetime.strptime(s,p)
            today=datetime.today()
            if dt>today: return "❌ Future DOB"
            age=today.year-dt.year-((today.month,today.day)<(dt.month,dt.day))
            if age<5: return "❌ Age too low (<5)"
            return "✅ Valid"
        except: continue
    return "❌ Invalid format"

def validate_aadhaar(num):
    if not num: return "❌ Missing"
    digits=re.sub(r"[^0-9]","",num)
    if len(digits)!=12: return "❌ Invalid length"
    return "✅ Valid" if verhoeff_check(digits) else "❌ Invalid (Verhoeff failed)"

# ---------- Load YOLO ----------
print("Loading YOLO model:",MODEL_PATH)
try: model=YOLO(MODEL_PATH); print("YOLO loaded")
except Exception as e: print("YOLO load failed:",e,file=sys.stderr); model=None

# ---------- Helpers ----------
def pil_to_cv2(image:Image.Image):
    arr=np.array(image.convert("RGB"))
    return cv2.cvtColor(arr,cv2.COLOR_RGB2BGR)

def open_input_image(input_obj):
    if isinstance(input_obj,str): return Image.open(input_obj)
    if hasattr(input_obj,"read"): return Image.open(io.BytesIO(input_obj.read()))
    if isinstance(input_obj,(bytes,bytearray)): return Image.open(io.BytesIO(input_obj))
    raise ValueError(f"Unsupported input type {type(input_obj)}")

# ---------- Core processing ----------
def process_image_file(file_path):
    pil_image=open_input_image(file_path)
    img=pil_to_cv2(pil_image)
    h,w=img.shape[:2]
    extracted={"aadhaar_number":"","dob":"","name":"","photo":""}
    if model is None: return {"error":"YOLO model not loaded."}
    results=model(img)
    for r in results:
        boxes=getattr(r,"boxes",None)
        if boxes is None: continue
        xyxy=boxes.xyxy.cpu().numpy()
        cls_ids=boxes.cls.cpu().numpy().astype(int)
        for (x1,y1,x2,y2),cls_id in zip(xyxy,cls_ids):
            x1,y1,x2,y2=map(int,[x1,y1,x2,y2])
            x1,y1=max(0,x1),max(0,y1)
            x2,y2=min(w-1,x2),min(h-1,y2)
            if x2<=x1 or y2<=y1: continue
            crop=img[y1:y2,x1:x2].copy()
            label=CLASS_MAP.get(cls_id,f"class_{cls_id}")
            text=tesseract_ocr(crop) if label!="photo" else ""
            if label=="aadhaar_number":
                digits=re.sub(r"[^0-9]","",text)
                if len(digits)==12: extracted["aadhaar_number"]=digits
            elif label in ("dob","name"):
                text=re.sub(r"[^A-Za-z0-9\-/ ]"," ",text).strip()
                if len(text)>len(extracted.get(label,"")): extracted[label]=text
            elif label=="photo":
                fname=f"photo_{uuid.uuid4().hex}.jpg"
                out_path=os.path.join(UPLOAD_FOLDER,fname)
                cv2.imwrite(out_path,crop)
                extracted["photo"]=out_path

    # Fallback OCR
    full_text=tesseract_ocr(img)
    if not extracted["aadhaar_number"]:
        m=re.search(r"\b\d{4}\s?\d{4}\s?\d{4}\b",full_text)
        if m: extracted["aadhaar_number"]=re.sub(r"\s+","",m.group(0))
    lines=[ln.strip() for ln in full_text.splitlines() if ln.strip()]
    if not extracted["dob"]:
        for ln in lines:
            m=re.search(r'(DOB|DoB|D\.O\.B|YOB|Year of Birth)[:\s]*([0-9]{2}[-/][0-9]{2}[-/][0-9]{4}|\d{4})',ln,re.I)
            if m: extracted["dob"]=m.group(2); break
    if not extracted["name"] and extracted["dob"]:
        for i,ln in enumerate(lines):
            if extracted["dob"] in ln and i>0:
                candidate=re.sub(r"[^A-Za-z\s\.]","",lines[i-1]).strip()
                if len(candidate)>=3: extracted["name"]=candidate; break

    validations={
        "aadhaar_number":validate_aadhaar(extracted.get("aadhaar_number")),
        "name":validate_name(extracted.get("name")),
        "dob":validate_dob(extracted.get("dob"))
    }

    # Annotated image
    annotated=img.copy()
    for r in results:
        boxes=getattr(r,"boxes",None)
        if boxes is None: continue
        xyxy=boxes.xyxy.cpu().numpy()
        cls_ids=boxes.cls.cpu().numpy().astype(int)
        for (x1,y1,x2,y2),cls_id in zip(xyxy,cls_ids):
            x1,y1,x2,y2=map(int,[x1,y1,x2,y2])
            color=(50,150,255)
            cv2.rectangle(annotated,(x1,y1),(x2,y2),color,2)
            cv2.putText(annotated,CLASS_MAP.get(cls_id,str(cls_id)),(x1,max(0,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

    ann_name=f"annotated_{uuid.uuid4().hex}.jpg"
    ann_path=os.path.join(UPLOAD_FOLDER,ann_name)
    cv2.imwrite(ann_path,cv2.cvtColor(annotated,cv2.COLOR_RGB2BGR))

    return {"extracted":extracted,"validations":validations,"annotated_path":ann_path,"photo_path":extracted.get("photo")}

# ---------- Flask ----------
app=Flask(__name__)
app.config["UPLOAD_FOLDER"]=UPLOAD_FOLDER

@app.route("/",methods=["GET"])
def index_route():
    return render_template("index.html")

@app.route("/scan",methods=["POST"])
def scan_route():
    if "file" not in request.files: return render_template("index.html",error="No file part")
    file=request.files["file"]
    if file.filename=="": return render_template("index.html",error="No file selected")
    if not allowed_file(file.filename): return render_template("index.html",error="Invalid file type")

    fname=f"upload_{uuid.uuid4().hex}.jpg"
    save_path=os.path.join(UPLOAD_FOLDER,fname)
    file.save(save_path)

    result=process_image_file(save_path)
    if result.get("error"): return render_template("index.html",error=result.get("error"))

    extracted=result["extracted"]
    validations=result["validations"]
    photo_rel=os.path.relpath(result.get("photo_path") or "",ROOT)

    return render_template("result.html",
        input_image=url_for('static',filename=f"uploads/{fname}"),
        extracted=extracted,
        validations=validations,
        photo_image=(url_for('static',filename=os.path.basename(photo_rel)) if photo_rel else None)
    )

if __name__=="__main__":
    app.run(host="0.0.0.0",port=7860)