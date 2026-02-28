# [PNEUMONIA DETECTION FROM CHEST X-RAY IMAGES USING DEEPLEARNING MODELS]


**Batch ID:** Batch-10  
**Course:** Undergrad Major Project 2026  
**Institution:** [PACE INSTITUTE OF TECHNOLOGY AND SCIENCES]

---

## 👥 Team Members
| J.ASSAn | 23KQ5A6105 |  |
| :--- | :--- | :--- |
| R.Susmitha | [22KQ1A6121] |  |
| P.Kavya | [22KQ1A6117] | @username |
| A.karthik | [22KQ1A6130] | @username |
 K.Nirmal Yeswanth | [22KQ1A6143 ] | @username |

---

## 🚀 Project Overview
**Problem Statement:** 

Pneumonia is one of the leading causes of respiratory-related deaths worldwide, especially among children and elderly patients. Manual interpretation of chest X-ray images requires experienced radiologists and may lead to delayed diagnosis in resource-limited healthcare environments.

This project proposes an Artificial Intelligence based automated diagnostic assistant capable of detecting pneumonia from chest X-ray images while also providing visual explanation using Grad-CAM, improving trust and interpretability.
---

**Key Objective:** 
The primary objective of this project is to develop an Explainable Artificial Intelligence (XAI) based pneumonia detection system using deep learning that can accurately classify chest X-ray images as Normal or Pneumonia while achieving high diagnostic performance (≈95%+ accuracy).

Additionally, the system aims to provide visual interpretation using Grad-CAM heatmaps, allowing users and medical practitioners to understand which lung regions influenced the model’s prediction, thereby improving transparency and trust in AI-assisted diagnosis.
---

## 📊 Dataset Information
* **Dataset Used:** Chest X-ray Pneumonia Dataset (Kermany Dataset)
Available at Kaggle:
* **Source:** [https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia]
* **Description:** The dataset contains pediatric chest X-ray images categorized into two classes: Normal and Pneumonia.
It includes approximately 5,800+ medical X-ray images organized into training, validation, and test folders.
The training set was used for model learning, while validation data monitored model performance during training.
Additionall,added 700 normal images for training for class imbalance.
We split 70-15-15 rule for better model learning.
---
* **Preprocessing:** 
>All chest X-ray images were resized to 224 × 224 pixels to match the ResNet50 input requirements.
>Images were normalized using ResNet50 preprocess_input() to improve model training stability.
>Data augmentation techniques such as rotation, zooming, shifting, and horizontal flipping were applied to the training dataset to improve generalization.
>Batch-wise image generators were used for efficient loading and GPU-based training in TensorFlow/Keras.
>Training, validation, and test datasets were kept separate to prevent data leakage during evaluation.
>The dataset contained properly labeled images, therefore no missing data handling was required.
* **Dataset Link:**  [https://drive.google.com/drive/folders/1FMSO0HNpe5z7Z6t_t1_-TPoI8LXXR025?usp=drive_link].
* **Model trained File**  [https://drive.google.com/file/d/1ap7uBFOPukiqPYAxKPdQQjVoau1KUJzY/view?usp=sharing]

## 🧠 Model Architecture & Methodology
* **Algorithm/Model:**
A ResNet50-based Convolutional Neural Network (CNN) with transfer learning was implemented to classify chest X-ray images as Normal or Pneumonia. The model was trained in two stages: Phase-1 feature extraction and Phase-2 fine tuning to improve medical feature learning.
* **Framework:**
The system was developed using TensorFlow and Keras in Google Colab with GPU support, including model evaluation using accuracy, ROC-AUC, confusion matrix analysis, and explainability through Grad-CAM visualization.


## 📈 Results & Performance
* **Project Results:**
| Metric | Value |
| :--- | :--- |
| Accuracy | 95.6% |
| Precision | 98.3% |
| Recall | 95.1% |
| F1-Score | 96.7% |
| ROC-AUC | 0.99 |

![Training History Graph](./docs/accuracy.png) 

![ROC Curve Graph](./roc/roccurve.png)

![Confusion matrix](./report/cm.png)



## 🔬 Explainable AI — Grad-CAM

To improve medical interpretability, Grad-CAM visualization is integrated.

Grad-CAM highlights:

🔴 Red / Yellow → Highly important infected lung regions

🟢 Blue / Green → Normal lung structure

This allows doctors to visually verify model reasoning.

Example Output:

Original X-ray → Model Prediction → Affected Lung Region Highlighted

This significantly improves clinical reliability.


🌐 Web Deployment

The trained model is deployed using:

Flask Framework

HTML5 + Bootstrap UI

Grad-CAM visualization pipeline

System Workflow:

Upload X-ray
      ↓
Model Prediction
      ↓
Confidence Score
      ↓
Grad-CAM Explanation
      ↓
Diagnostic Report
🚀 Installation

Clone repository:

git clone https://github.com/PACE-ITS/major-project-batch-10

Install dependencies:

pip install -r requirements.txt

Run application:

python app.py

Open browser:

http://127.0.0.1:5000
📷 Project Output


🔮 Future Improvements

Multi-disease lung detection

RSNA dataset generalization

Cloud deployment

Doctor feedback integration


