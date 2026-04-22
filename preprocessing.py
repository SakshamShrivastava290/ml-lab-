"""
=============================================================
  DATA PREPROCESSING — Standalone Lab Program
=============================================================
  Steps: Missing values, Encoding, Scaling, Train-Test Split
  Libraries: pandas, numpy, sklearn
  Usage: python preprocessing.py
=============================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def run():
    # --- Load Dataset ---
    path = input("Enter CSV file path (default: data.csv): ").strip() or "data.csv"
    df = pd.read_csv(path)
    print(f"\n✅ Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\n--- Missing Values Before ---\n{df.isnull().sum()}\n")

    # --- Step 1: Handle Missing Values ---
    print("STEP 1: Handling Missing Values")
    for col in df.columns:
        missing = df[col].isnull().sum()
        if missing > 0:
            if df[col].dtype in ['float64', 'int64']:
                fill = df[col].mean()
                df[col].fillna(fill, inplace=True)
                print(f"  '{col}': filled {missing} nulls with mean = {fill:.2f}")
            else:
                fill = df[col].mode()[0]
                df[col].fillna(fill, inplace=True)
                print(f"  '{col}': filled {missing} nulls with mode = '{fill}'")
    print(f"  Total missing after: {df.isnull().sum().sum()}\n")

    # --- Step 2: Encode Categorical Variables ---
    print("STEP 2: Encoding Categorical Variables")
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    if cat_cols:
        for col in cat_cols:
            le = LabelEncoder()
            original = df[col].unique()
            df[col] = le.fit_transform(df[col].astype(str))
            print(f"  '{col}': {list(original)} → {list(le.transform(le.classes_))}")
    else:
        print("  No categorical columns found.")
    print()

    # --- Step 3: Feature Scaling ---
    print("STEP 3: Feature Scaling (StandardScaler)")
    target = input("Enter target column name: ").strip()
    X = df.drop(columns=[target])
    y = df[target]
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    print(f"  Scaled {X.shape[1]} features")
    print(f"  Mean ≈ 0 : {X_scaled.mean().values.round(4)}")
    print(f"  Std  ≈ 1 : {X_scaled.std().values.round(4)}\n")

    # --- Step 4: Train-Test Split ---
    print("STEP 4: Train-Test Split (80/20)")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    print(f"  Training set : {X_train.shape[0]} samples")
    print(f"  Testing set  : {X_test.shape[0]} samples")

    print(f"\n✅ Preprocessing complete!")
    print(f"\n--- Preprocessed Data (first 5 rows) ---\n{X_train.head()}")


if __name__ == "__main__":
    run()
