# Earnings Manipulation Detection Using AdaBoost

## 📌 Overview

This project focuses on detecting earnings manipulation using machine learning methods. We leverage publicly available financial statement data to develop an AdaBoost ensemble model that classifies firms as manipulators or non-manipulators. The analysis is inspired by the Beneish M-Score framework but enhanced using modern classification techniques.

---

## 🎯 Objectives

- Build an AdaBoost model to classify companies as manipulators or non-manipulators.
- Preprocess raw financial data including imputation and scaling.
- Tune hyperparameters to optimize classification performance.
- Evaluate model using classification metrics, ROC curve, and PR curve.

---

## 🧠 Methodology

### Data Source
- **IMB579-XLS-ENG.xlsx**
  - Sheet 1: Sample for Model Development
  - Sheet 2: Complete Data for Final Testing

### Workflow

1. **Preprocessing**
   - Dropped irrelevant columns (e.g., Company ID)
   - Imputed missing values using median strategy
   - Standardized features using `StandardScaler`

2. **Model Development**
   - Used `AdaBoostClassifier` with `DecisionTreeClassifier` as the base estimator.
   - Pipeline included both preprocessing and classification steps.
   - Grid search performed for hyperparameter tuning (e.g., `n_estimators`, `learning_rate`, `max_depth`).

3. **Model Evaluation**
   - Accuracy, precision, recall, and F1-score
   - Cross-validation on training data
   - Threshold tuning for F1-score optimization
   - ROC and Precision-Recall curves on full dataset

---

## 📊 Results

- **Cross-validation accuracy (after tuning):** ~0.81
- **Best threshold based on F1-score:** ~0.41
- **ROC-AUC:** ~0.86 on complete dataset
- Confusion matrix and classification reports confirm improved precision-recall trade-off post tuning.

---

## 🧰 Requirements

Install dependencies using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
python MFin_704_Adaboost_Final.py
```

## 🙋 Author

**Jinyan Yong**
 Master of Finance – McMaster University