# 🦷 Dental Disease Diagnostic System (DentalAI)

---
## Demo
<!-- ![Teeth Classification Demo](results/demo/teeth_classification.gif) -->
---

## Project Overview

This project is an end-to-end medical image classification system, designed to assist dentists in **identifying 7 different oral conditions** with high precision. This project was developed as part of my internship at **Cellula**.

---

## 🛠️ Project Evolution
🔹 **Week 1: Foundation & Custom Modeling**
- **Custom CNN Architecture:** Developed a baseline model from scratch.

- **Data Augmentation:** Applied spatial transformations to handle dataset variance.

- **CLAHE Preprocessing:** Implemented Contrast Limited Adaptive Histogram Equalization to enhance dental features.

- **Baseline Deployment:** Initially tested using a Flask web interface.

🔹 **Week 2: Advanced Optimization & Professional Deployment**
- **Transfer Learning:** Implemented EfficientNet-B0 (Pre-trained on ImageNet) for superior feature extraction.

- **Streamlit Dashboard:** Built a professional-grade medical dashboard for real-time diagnostics.

## 📊 Performance Comparison


| Stage |        Model       |  Preprocessing  | Test Accuracy |
|-------|--------------------|-----------------|---------------|
| Week1 |    Baseline CNN    |   Normalization |    ~96.8%     |
| Week1 |    Augmented CNN   |   Augmentation  |    ~98.9%     |
| Week1 |    CLAHE Model     |   CLAHE + Aug   |    ~96.2%     |
| Week2 |   EfficientNet-B0  |     Transfer    |    ~99.2%     |

---

##  🔬 Technologies Used

- **Core:** Python, TensorFlow, Keras.

- **Image Processing:** OpenCV (CLAHE), PIL.

- **Deployment:** `Streamlit` (Current), Flask (Legacy).

- **Visualization:** Plotly Express, Seaborn (Confusion Matrix).

---

## 🚀 How to Run the Project

### 1️⃣ Install Requirements
```bash
cd teeth_app_streamlit
pip install -r requirements.txt
```

### 2️⃣ Run Streamlit App
```bash
streamlit run app.py
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

## 📌 Key Takeaways
1. **CLAHE** was crucial for highlighting subtle dental symptoms that the model would otherwise miss.

2. **EfficientNet** provided the most stable performance and faster convergence compared to the custom architecture.

## 👩‍💻 Author
**Alaa Sayed** - AI Engineer
