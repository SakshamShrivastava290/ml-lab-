"""
=============================================================
  SUPPORT VECTOR MACHINE (SVM) — Standalone Lab Program
=============================================================
  Supports: Classification (SVC) AND Regression (SVR)
  Classification Metrics: Accuracy, Precision, Recall, F1, Confusion Matrix
  Regression Metrics: MSE, MAE, R² Score
  Libraries: pandas, numpy, matplotlib, seaborn, sklearn
  Usage: python svm.py
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix,
                             mean_squared_error, r2_score, mean_absolute_error)


def load_and_preprocess(path, target_col):
    df = pd.read_csv(path)
    print(f"✅ Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['float64', 'int64']:
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        print(f"  Encoded '{col}'")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"  Train: {X_train.shape[0]} | Test: {X_test.shape[0]}\n")
    return X_train, X_test, y_train, y_test


def evaluate_classification(y_true, y_pred, model_name):
    print(f"\n{'=' * 50}")
    print(f"  📊 {model_name} — Classification Results")
    print(f"{'=' * 50}")
    print(f"  Accuracy  : {accuracy_score(y_true, y_pred):.4f}")
    print(f"  Precision : {precision_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"  Recall    : {recall_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"  F1 Score  : {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"{'=' * 50}")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} — Confusion Matrix')
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.tight_layout(); plt.show()


def evaluate_regression(y_true, y_pred, model_name):
    print(f"\n{'=' * 50}")
    print(f"  📈 {model_name} — Regression Results")
    print(f"{'=' * 50}")
    print(f"  MSE  : {mean_squared_error(y_true, y_pred):.4f}")
    print(f"  MAE  : {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"  R²   : {r2_score(y_true, y_pred):.4f}")
    print(f"{'=' * 50}")
    plt.figure(figsize=(7, 5))
    plt.scatter(y_true, y_pred, color='steelblue', edgecolors='k', alpha=0.7)
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], color='tomato', linewidth=2, linestyle='--', label='Ideal Fit')
    plt.xlabel('Actual'); plt.ylabel('Predicted')
    plt.title(f'{model_name} — Actual vs Predicted')
    plt.legend(); plt.tight_layout(); plt.show()


def run():
    print("\n" + "=" * 60)
    print("  🔷 SUPPORT VECTOR MACHINE (SVM)")
    print("=" * 60)

    path = input("Enter CSV path (default: data.csv): ").strip() or "data.csv"
    target = input("Enter target column: ").strip()

    # Ask user: classification or regression
    mode = input("Classification or Regression? (c/r): ").strip().lower()

    X_train, X_test, y_train, y_test = load_and_preprocess(path, target)

    if mode == 'r':
        model = SVR(kernel='rbf')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        evaluate_regression(y_test, y_pred, "SVR (RBF Kernel)")
    else:
        model = SVC(kernel='rbf', random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        evaluate_classification(y_test, y_pred, "SVC (RBF Kernel)")


if __name__ == "__main__":
    run()
