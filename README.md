---
title: Aadhaar Validator
emoji: 🪪
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
license: apache-2.0
---
# Aadhaar Validator 🔎

A Flask web app that validates Aadhaar cards using YOLO detection + OCR.  
It extracts Aadhaar Number, Name, DOB, and displays results neatly in a bordered card.  

## 🚀 Features
- Upload Aadhaar image → detect Aadhaar number, name, DOB
- YOLO model for detection (`best.pt`)
- Extracted data shown in a professional bordered card
- Cropped Aadhaar photo displayed neatly
- SweetAlert pop-up ("Scanning Aadhaar...")
- Clean UI with iframe embedding (hides Hugging Face navbar)

---

## 📂 Project Structure
See the tree above.

---

## ⚙️ Setup

### 1. Clone the repo
```bash
git clone https://huggingface.co/spaces/username/aadhaar-validator
cd aadhaar-validator