"""
=============================================================
  RANDOM FOREST — Standalone Lab Program
=============================================================
  Supports: Classification AND Regression
  Libraries: pandas, numpy, matplotlib, seaborn, sklearn
  Usage: python random_forest.py
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
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


def evaluate_classification(y_true, y_pred, name):
    print(f"\n{'='*50}\n  📊 {name} — Classification Results\n{'='*50}")
    print(f"  Accuracy  : {accuracy_score(y_true, y_pred):.4f}")
    print(f"  Precision : {precision_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"  Recall    : {recall_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"  F1 Score  : {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} — Confusion Matrix')
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.tight_layout(); plt.show()


def evaluate_regression(y_true, y_pred, name):
    print(f"\n{'='*50}\n  📈 {name} — Regression Results\n{'='*50}")
    print(f"  MSE : {mean_squared_error(y_true, y_pred):.4f}")
    print(f"  MAE : {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"  R²  : {r2_score(y_true, y_pred):.4f}")
    plt.figure(figsize=(7, 5))
    plt.scatter(y_true, y_pred, color='steelblue', edgecolors='k', alpha=0.7)
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], 'r--', lw=2, label='Ideal Fit')
    plt.xlabel('Actual'); plt.ylabel('Predicted')
    plt.title(f'{name} — Actual vs Predicted')
    plt.legend(); plt.tight_layout(); plt.show()


def run():
    print("\n" + "=" * 60)
    print("  🌳 RANDOM FOREST")
    print("=" * 60)
    path = input("Enter CSV path (default: data.csv): ").strip() or "data.csv"
    target = input("Enter target column: ").strip()
    mode = input("Classification or Regression? (c/r): ").strip().lower()
    X_train, X_test, y_train, y_test = load_and_preprocess(path, target)
    n = input("Number of trees (default 100): ").strip()
    n = int(n) if n else 100
    if mode == 'r':
        model = RandomForestRegressor(n_estimators=n, random_state=42)
        model.fit(X_train, y_train)
        evaluate_regression(y_test, model.predict(X_test), f"Random Forest Regressor (n={n})")
    else:
        model = RandomForestClassifier(n_estimators=n, random_state=42)
        model.fit(X_train, y_train)
        evaluate_classification(y_test, model.predict(X_test), f"Random Forest Classifier (n={n})")


if __name__ == "__main__":
    run()
