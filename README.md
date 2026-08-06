# Medical Diagnostic AI: Breast Cancer Classification & Performance Evaluation

This project implements an end-to-end Machine Learning pipeline for binary classification on the Scikit-Learn Breast Cancer Wisconsin dataset. It focuses on going beyond standard accuracy to evaluate models using business-critical metrics like Precision, Recall, F1-Score, and ROC-AUC.

## 📌 Project Overview
In medical diagnostic applications, minimizing False Negatives is critical. This repository demonstrates how to train, evaluate, and tune binary classifiers while addressing real-world challenges such as class imbalance and precision-recall trade-offs.

## 🛠️ Key Steps Implemented
1. **Data Preprocessing & Stratification:** Split data preserving target distribution and scaled features using `StandardScaler`[cite: 1].
2. **Baseline Model Training:** Implemented Logistic Regression as a baseline classifier[cite: 1].
3. **Comprehensive Metrics Analysis:** Evaluated model performance using Confusion Matrices, Precision, Recall, and F1-Score[cite: 1].
4. **ROC-AUC Visualization:** Plotted ROC curves to assess true positive vs. false positive trade-offs[cite: 1].
5. **Class Imbalance Handling:** Applied balanced class weighting (`class_weight='balanced'`) to optimize recall for malignant predictions[cite: 1].
6. **Model Comparison:** Compared Logistic Regression performance against an unconstrained Decision Tree Classifier[cite: 1].

## 📊 Key Results

| Model | Accuracy | Precision (Malignant) | Recall (Malignant) | F1-Score (Malignant) | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 98% | 0.98 | 0.98 | 0.98 | 0.9974 |
| **Logistic Regression (Balanced)** | 98% | 0.98 | 0.98 | 0.98 | 0.9974 |
| **Decision Tree Classifier** | 94% | 0.91 | 0.93 | 0.92 | — |

## 💡 Technical Insights
* **Metric Selection:** Recall is prioritized over Precision in medical screening to prevent missing actual malignant cases (False Negatives)[cite: 1].
* **Imbalance Handling:** F1-Score and class-weighted loss functions provide a far more robust evaluation than accuracy when classes are imbalanced[cite: 1].
* **Model Choice:** Logistic Regression demonstrated higher overall stability, superior generalization, and higher ROC-AUC compared to the Decision Tree[cite: 1].
