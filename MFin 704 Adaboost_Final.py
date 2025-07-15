#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 10:21:34 2025

@author: jinyanyong
"""

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from pandas.plotting import scatter_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc, precision_recall_curve, confusion_matrix
from sklearn.model_selection import TunedThresholdClassifierCV

# %% Import data from excel for EDA
file_path = "IMB579-XLS-ENG.xlsx"  
df = pd.read_excel(file_path, sheet_name="Complete Data")

# Drop irrelevant columns
# remove the company ID because it is useless
df = df.drop(columns=["Company ID"], errors="ignore")

# Convert categorical 'Manipulator' column to binary (Yes → 1, No → 0)
df["Manipulator"] = df["Manipulator"].map({"Yes": 1, "No": 0})


print(df.head())
print(df.info())

# %% check if null values exist
if df.isna().sum().sum() == 0:
    print("No missing values")
else:
    print("Dataset contains missing values")

print(df.isnull().sum())


# %% Plotting Scatter check for correlation? Should we differentiate by Class
sns.pairplot(df, hue='C-Manipulator'.upper(), diag_kind='kde')

#Plot histogram
df.hist(bins=50, figsize=(12, 8))
plt.show()


#%%
dfcor = df.drop(["Manipulator","C-MANIPULATOR"],axis=1)
corr_matrix = dfcor.corr()
# heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.show()
# %% Import data from excel for AdaBoost

# Read the Excel file and select the “Sample Data” sheet
file_path = "IMB579-XLS-ENG.xlsx"  
df = pd.read_excel(file_path, sheet_name="Sample for Model Development")
df["Manipulator"] = df["Manipulator"].map({"Yes": 1, "No": 0})

# %% Select Features (remove Company ID and non-numeric columns)
X = df.drop(["Company ID", "Manipulator", "C-MANIPULATOR"], axis=1)
y = df["C-MANIPULATOR"]  # Target variable

#%% Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=df["C-MANIPULATOR"], random_state=42
)

#%% Create a pipeline for the model
data_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),  # # Missing value filling
    ("scaler", StandardScaler())  # Data standardization
])

#%% Create an AdaBoost pipeline
adaboost_pipeline = Pipeline([
    ("preprocessing", data_pipeline),
    ("classifier", AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2),
                                      n_estimators=200, random_state=42))
])

#%% Fit the model
adaboost_pipeline.fit(X_train, y_train)

#%% Predict the test set
y_pred = adaboost_pipeline.predict(X_test)

#%% Evaluate the model
print("=== Adaboost report ===")
print(classification_report(y_test, y_pred))

#%% Cross Validation
cv_scores = cross_val_score(adaboost_pipeline, X_train, y_train, cv=5, scoring="accuracy")
print(f"Initial Cross Validation Accuracy: {cv_scores.mean():.4f}")

#%% Use parameter grid for GridSearchCV
param_grid = {
    "classifier__n_estimators": [50, 100, 200, 300], # number of estimators
    "classifier__learning_rate": [0.01, 0.1, 0.5, 1], # learning rate
    "classifier__estimator__max_depth": [1, 2, 3],
    "classifier__estimator__min_samples_split": [2, 5, 10],
    "classifier__estimator__min_samples_leaf": [1, 2, 5]
}
grid_search = GridSearchCV(adaboost_pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_

#%% Cross Validation
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring="accuracy")
print(f"Cross Validation Accuracy: {cv_scores.mean():.4f}")

#%% Threshold Tuning and Evaluation
y_probs_test_sample = best_model.predict_proba(X_test)[:, 1]  # Extract probability of class 1

# Compute the Precision-Recall curve on Sample Data test set
precision, recall, thresholds = precision_recall_curve(y_test, y_probs_test_sample)

# Compute F1-score
f1_scores = np.divide(2 * (precision * recall), (precision + recall), 
                      out=np.zeros_like(precision), where=(precision + recall) != 0)

# Ensure correct threshold indexing
best_threshold = thresholds[np.argmax(f1_scores[:-1])]  
print(f"Best threshold based on F1-score (Sample Data test set): {best_threshold:.3f}")

# Reclassify the Sample Data test set using the new threshold
y_pred_final_sample = (y_probs_test_sample >= best_threshold).astype(int)

# Evaluate the model on Sample Data test set after threshold tuning
print("=== Classification Report (Sample Data test set after Threshold Tuning) ===")
print(classification_report(y_test, y_pred_final_sample))
cm = confusion_matrix(y_test, y_pred_final_sample)
print("Confusion Matrix:\n", cm)

#%%
# Read the Excel file and select the “Complete Data” sheet
df_complete = pd.read_excel("IMB579-XLS-ENG.xlsx", sheet_name="Complete Data")
df_complete["Manipulator"] = df_complete["Manipulator"].map({"Yes": 1, "No": 0})

X_test_complete = df_complete.drop(["Company ID", "Manipulator", "C-MANIPULATOR"], axis=1)
y_test_complete = df_complete["C-MANIPULATOR"]

#%% Obtain probability predictions for Complete Data
y_probs_complete = best_model.predict_proba(X_test_complete)[:, 1]

# **No Threshold Tuning on Complete Data**
y_pred_complete = best_model.predict(X_test_complete)

# Evaluate Final Model on Complete Data
print("=== Classification Report on Complete Data ===")
print(classification_report(y_test_complete, y_pred_complete))

# Confusion Matrix for Complete Data
cm_complete = confusion_matrix(y_test_complete, y_pred_complete)
print("Confusion Matrix:\n", cm_complete)

#%% ROC Curve for Complete Data
fpr, tpr, _ = roc_curve(y_test_complete, y_probs_complete)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve on Complete Data")
plt.legend()
plt.show()

#%% Precision-Recall Curve for Complete Data
precision, recall, _ = precision_recall_curve(y_test_complete, y_probs_complete)
plt.figure(figsize=(8,6))
plt.plot(recall, precision, color='blue', lw=2, label="Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve on Complete Data")
plt.legend()
plt.show()

#%% Compute train and test accuracy correctly
train_acc = accuracy_score(y_train, best_model.predict(X_train))
test_acc = accuracy_score(y_test, best_model.predict(X_test))
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")


