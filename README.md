---
title: Aadhaar Validator
emoji: 💻
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: app.py
pinned: true
license: apache-2.0
---

# Aadhaar Card Verification System

This Space verifies Aadhaar card details using **YOLOv8 (Ultralytics)** for detection and **Tesseract OCR** for text extraction.  
It validates:
- Aadhaar Number (with Verhoeff checksum)  
- Name  
- Date of Birth  
- Extracts Aadhaar Photo  

## How it Works
1. Upload an Aadhaar card image (`.jpg`, `.jpeg`, `.png`).  
2. The system detects regions (name, DOB, Aadhaar number, photo).  
3. OCR extracts text.  
4. Aadhaar number is validated with Verhoeff checksum.  
5. Results and detected photo are displayed.  

## Run Locally
```bash
pip install -r requirements.txt
python app.py