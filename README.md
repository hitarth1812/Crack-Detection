# Crack-Detection
This is a professional, high-quality `README.md` file specifically designed for a GitHub repository. It includes badges, clear sections, and placeholders for your specific results.

---

# AI-Based Concrete Crack Detection

An end-to-end deep learning system for automatic detection of cracks in concrete surfaces using **EfficientNetB0** with transfer learning and **Grad-CAM** explainability. This project achieves **99.72% accuracy**, making it a reliable tool for civil infrastructure inspection.

##  Project Overview

Manual crack detection in civil structures is time-consuming, subjective, and difficult to scale. This project automates the process using Computer Vision to ensure structural safety.

* **Goal:** Classify images into 'Crack' or 'No Crack'.
* **Methodology:** Transfer Learning using EfficientNetB0.
* **Interpretability:** Grad-CAM heatmaps to visualize exactly where the model "looks" to identify a crack.

---

## Key Features

* **High Accuracy:** Near-perfect 99.72% test accuracy.
* **Efficiency:** Uses EfficientNetB0 for optimal balance between speed and performance.
* **Explainable AI (XAI):** Integrated Grad-CAM for model transparency.
* **Automated Pipeline:** Handles everything from data preprocessing to final evaluation.

---

## Model Architecture

The model utilizes the **EfficientNetB0** backbone, pre-trained on ImageNet, with custom top layers for binary classification.

| Layer | Type | Specifications |
| --- | --- | --- |
| **Backbone** | EfficientNetB0 | Pre-trained ImageNet weights |
| **Pooling** | GlobalAveragePooling2D | Reduces spatial dimensions |
| **Regularization** | Dropout | Rate = 0.5 |
| **Output Layer** | Dense | 1 Neuron (Sigmoid) |

### Optimization Details

* **Loss Function:** Binary Cross-Entropy

* **Optimizer:** Adam ()
* **Input Shape:** 

---

## Dataset

The dataset consists of high-resolution images of concrete surfaces categorized into:

1. **Positive (Crack)**
2. **Negative (No Crack)**

**Data Split:**

* **Training:** 70%
* **Validation:** 15%
* **Testing:** 15%

---

##  Explainability with Grad-CAM

Gradient-weighted Class Activation Mapping (Grad-CAM) is used to validate the model's decision-making process. By visualizing the gradients flowing into the final convolutional layer, we produce a heatmap that highlights the "crack" regions.

> **Why it matters:** In civil engineering, knowing *why* a model flagged a structure is as important as the flag itself.

---

##  Tech Stack

* **Language:** Python
* **Deep Learning:** TensorFlow / Keras
* **Image Processing:** OpenCV
* **Data Analysis:** NumPy, Pandas, Scikit-learn
* **Visualization:** Matplotlib, Seaborn

---


##  Results

The model demonstrates exceptional performance across all metrics:

| Metric | Score |
| --- | --- |
| **Test Accuracy** | 99.72% |
| **Precision** | ~1.00 |
| **Recall** | ~1.00 |

---

##  Authors

**Hitarth Khatiwala** *Dept. of Artificial Intelligence & Machine Learning* [24aiml019@charusat.edu.in](mailto:24aiml019@charusat.edu.in)

---


