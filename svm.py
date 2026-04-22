"""
=============================================================
  SUPPORT VECTOR MACHINE (SVM) — Standalone Lab Program
=============================================================
  Metrics: Accuracy, Precision, Recall, F1 Score
  Visualization: Confusion Matrix Heatmap
  Libraries: pandas, matplotlib, seaborn, sklearn
  Usage: python svm.py
=============================================================
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


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
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()


def run():
    print("\n" + "=" * 60)
    print("  🔷 SUPPORT VECTOR MACHINE (SVM)")
    print("=" * 60)

    path = input("Enter CSV path (default: data.csv): ").strip() or "data.csv"
    target = input("Enter target column: ").strip()
    X_train, X_test, y_train, y_test = load_and_preprocess(path, target)

    # Train with RBF kernel
    model = SVC(kernel='rbf', random_state=42)
    model.fit(X_train, y_train)

    # Predict & Evaluate
    y_pred = model.predict(X_test)
    evaluate_classification(y_test, y_pred, "SVM (RBF Kernel)")


if __name__ == "__main__":
    run()
