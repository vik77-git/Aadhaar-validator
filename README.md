---
title: Aadhaar Validator
emoji: 🪪
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---

# 🪪 Aadhaar Card Verification System (Flask)

This Space is a **Flask-based web application** for validating Aadhaar cards using **YOLOv8 + Tesseract OCR**.  
It automatically detects Aadhaar number, name, date of birth, and photo, then performs validation checks like **Verhoeff algorithm** for Aadhaar number and DOB correctness.

---

## 🚀 Features
- Upload Aadhaar card image (`.jpg`, `.jpeg`, `.png`)
- YOLOv8 model detects Aadhaar card fields (number, name, DOB, photo)
- OCR (Tesseract) extracts text from detected fields
- Aadhaar number validated with **Verhoeff algorithm**
- DOB validated for format, non-future, and age > 5 years
- Extracted **photo crop** displayed separately
- Professional responsive UI (HTML + CSS)

---

## 🛠️ Project Structure