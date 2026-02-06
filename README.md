# 🦷 Teeth Image Classification System

This project is an **Image Classification system for dental images** using Deep Learning.
It compares multiple preprocessing techniques and CNN models to achieve the best classification performance.

---
## Demo
results\demo\teeth_classification.gif
---

## Project Overview

The system classifies teeth images into **7 different classes** using:
- A baseline CNN model
- Data Augmentation
- CLAHE (Contrast Limited Adaptive Histogram Equalization)

The final selected model is deployed using **Flask** with a simple frontend interface.\

---

## Models Used

- Custom CNN Architecture
- Batch Normalization
- Dropout for regularization
- Early Stopping & Learning Rate Reduction

### Preprocessing Techniques:
- Normalization
- Data Augmentation
- CLAHE (Contrast enhancement)

---

## Model Performance (Test Set)

| Model | Test Accuracy |
|------|---------------|
| Baseline | ~96.8% |
| Augmented | ~98.9%** |
| CLAHE | ~96.2% |

The **Baseline model** was selected for deployment.

---

## Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- Flask
- HTML, CSS

---

## Features

- Upload dental image
- Predict tooth class
- Display prediction confidence
- Clean and simple UI

---

## 🚀 How to Run the Project

### 1️⃣ Install Requirements
```bash
pip install -r requirements.txt
```

### 2️⃣ Run Flask App
```bash
python app.py
```

### 3️⃣ Open in Browser
```bash
http://127.0.0.1:5000
```

## Dataset

- Images are organized into folders by class

- Dataset split into:

    - Training

    - Validation

    - Testing

## 📌 Notes

CLAHE improves contrast but may cause over-enhancement if not tuned properly.

Data Augmentation improved generalization significantly.

## Author

Developed by Alaa Sayed
