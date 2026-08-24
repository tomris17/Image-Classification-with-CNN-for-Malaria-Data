# Malaria Cell Image Classification: Custom CNN vs. Transfer Learning

An end-to-end Computer Vision project comparing a custom **Convolutional Neural Network (CNN)** against **Transfer Learning architectures (VGG16 / MobileNetV2)** to detect malaria parasites in microscopic blood smear images.

---

##  Project Overview
Malaria diagnosis traditionally relies on manual microscopic evaluation of blood smears, which is time-consuming and prone to human error. This project automates the detection pipeline using deep learning:

- Data cleaning, format filtering, and normalization of microscopic cell images.
- Training a custom multi-layer CNN with **Dropout** and **EarlyStopping** to eliminate overfitting.
- Implementing **Transfer Learning** using pre-trained ImageNet models (`VGG16`, `MobileNetV2`) for feature extraction.
- Benchmarking model performances via Accuracy/Loss curves, Confusion Matrices, and Classification Reports.

---

##  Dataset
The dataset consists of **27,558 single-cell images** evenly split across two classes:
- **Parasitized:** Infected red blood cells containing distinct staining artifacts and ring-form trophozoites.
- **Uninfected:** Healthy red blood cells with homogeneous cytoplasm.

Please get this Dataset from the official NIH Website: https://ceb.nlm.nih.gov/repositories/malaria-datasets/

or

https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria

---

##  Tech Stack
- **Languages:** Python 3.x
- **Frameworks:** TensorFlow / Keras
- **Image Processing:** OpenCV (`cv2`)
- **Data & Scientific Computing:** NumPy, Pandas, Scikit-Learn
- **Visualization:** Matplotlib, Seaborn

---

## Model Architectures

### 1. Custom CNN
- **Input:** `(170, 170, 3)`
- **Feature Extraction:** 3x `[Conv2D (ReLU) + MaxPooling2D]` blocks (32, 64, 64 filters)
- **Classifier Head:** `Flatten` $\rightarrow$ `Dense(64, ReLU)` $\rightarrow$ `Dropout(0.5)` $\rightarrow$ `Dense(1, Sigmoid)`
- **Optimization:** Adam optimizer, Binary Cross-Entropy loss

### 2. Transfer Learning (Pre-trained on ImageNet)
- **VGG16 / MobileNetV2:** Base models used as frozen feature extractors (`trainable = False`).
- **Classification Head:** `GlobalAveragePooling2D` $\rightarrow$ `Dense(64, ReLU)` $\rightarrow$ `Dropout(0.5)` $\rightarrow$ `Dense(1, Sigmoid)`

---

##  Performance & Results

| Model Architecture | Validation Accuracy | Validation Loss | Key Characteristics |
| :--- | :---: | :---: | :--- |
| **Custom CNN** | **~94.5%** | **~0.17** | Fast convergence, low computational cost, highly accurate |
| **Transfer Learning (VGG16 / MobileNetV2)** | **~94.0 - 95.0%** | **~0.15 - 0.18** | Robust feature representations, high generalizability |

- **Overfitting Prevention:** Incorporating `Dropout(0.5)` and `EarlyStopping(monitor='val_loss', patience=2)` prevented metric divergence, ensuring training and validation curves remained aligned.

---

##  Getting Started

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/malaria-cell-classification.git](https://github.com/your-username/malaria-cell-classification.git)
cd malaria-cell-classification
