# 🥔 Potato Disease Detection using Custom CNN

A deep learning–based image classification project built with **TensorFlow** and **Keras** to detect **potato leaf diseases**.  
This model classifies leaf images into three categories — **Early Blight**, **Late Blight**, and **Healthy** — using a **custom Convolutional Neural Network (CNN)** trained from scratch on the **PlantVillage** dataset.

---

## 🚀 Project Overview
This project demonstrates a complete deep learning pipeline for **plant disease diagnosis**.  
It includes:
- Dataset preparation and augmentation  
- Custom CNN model design  
- Training with callbacks (checkpointing, learning rate reduction, early stopping)  
- Visualization of accuracy/loss  
- Model evaluation, saving, and inference  

The goal is to assist farmers and researchers by automating the identification of potato diseases directly from leaf images.

---

## 📁 Dataset
- **Source:** [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
- **Subset used:** Potato leaf images (3 classes)
  - `Potato___Early_blight`
  - `Potato___Late_blight`
  - `Potato___Healthy`
- **Total Images:** 2,152  
  - 1,722 for training  
  - 430 for validation  

---

## 🧠 Model Architecture
A **custom CNN** built manually using Keras Sequential API.

| Layer Type | Parameters / Notes |
|-------------|--------------------|
| Input | (224 × 224 × 3) |
| Data Augmentation | Random flip, rotation, zoom, contrast |
| Conv2D(32, 3×3) + MaxPool(2×2) | Extract low-level features |
| Conv2D(64, 3×3) + MaxPool(2×2) | Deeper feature extraction |
| Conv2D(128, 3×3) + MaxPool(2×2) | Higher-level feature maps |
| GlobalAveragePooling2D | Reduce spatial dimensions |
| Dense(128, ReLU) + Dropout(0.3) | Fully connected layer |
| Dense(3, Softmax) | Output probabilities for each class |

**Total Parameters:** ~110K (lightweight & efficient)

---

## ⚙️ Training Details
- **Optimizer:** Adam (`lr = 1e-4`)  
- **Loss:** Sparse Categorical Crossentropy  
- **Metrics:** Accuracy  
- **Batch Size:** 32  
- **Epochs:** 30  
- **Validation Split:** 20%  
- **Callbacks:**  
  - ModelCheckpoint (save best model)  
  - ReduceLROnPlateau (adaptive learning rate)  
  - EarlyStopping (avoid overfitting)

Training performed on **Google Colab GPU** (T4).

---

## 📈 Results

| Metric | Training | Validation |
|---------|-----------|-------------|
| **Accuracy** | 89.1% | **74.88%** |
| **Loss** | 0.34 | **0.63** |

- The model generalizes well on unseen validation data.
- Minor overfitting observed, which is expected for small datasets.
- Validation accuracy stabilizes around 75% after ~20 epochs.

**Training curves:**
- 📈 Accuracy: steady improvement and stabilization  
- 📉 Loss: consistent decrease during training  

---

## 🧩 Model Saving and Inference
The best model was saved automatically as **`potato_model.keras`** (native Keras format).  
Class labels were stored in `labels.json`.

### 🔍 Example Inference
```python
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np, json

# Load model and labels
model = keras.models.load_model("final_model/potato_model.keras")
with open("final_model/labels.json") as f:
    class_names = json.load(f)

# Predict on a single image
def infer_from_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    arr = np.expand_dims(image.img_to_array(img) / 255.0, axis=0)
    preds = model.predict(arr)
    idx = np.argmax(preds[0])
    print(f"Predicted: {class_names[idx]} ({preds[0][idx]:.2f})")

infer_from_image("test_leaf.jpg")
