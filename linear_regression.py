"""
=============================================================
  LINEAR REGRESSION — Standalone Lab Program
=============================================================
  Covers: Simple Linear Regression (single feature)
  Metrics: MSE, MAE, R² Score
  Visualization: Regression line plot
  Libraries: pandas, numpy, matplotlib, sklearn
  Usage: python linear_regression.py
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def load_and_preprocess(path, target_col):
    """Load CSV, handle missing values, encode categoricals, scale features."""
    df = pd.read_csv(path)
    print(f"✅ Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

    # Handle missing values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['float64', 'int64']:
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)

    # Encode categorical variables
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        print(f"  Encoded '{col}'")

    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Scale features
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"  Train: {X_train.shape[0]} | Test: {X_test.shape[0]}\n")
    return X_train, X_test, y_train, y_test


def run():
    print("\n" + "=" * 60)
    print("  📈 SIMPLE LINEAR REGRESSION")
    print("=" * 60)

    path = input("Enter CSV path (default: data.csv): ").strip() or "data.csv"
    target = input("Enter target column: ").strip()

    X_train, X_test, y_train, y_test = load_and_preprocess(path, target)

    # Select one feature for simple regression
    print(f"Available features: {list(X_train.columns)}")
    feat = input("Enter feature for simple regression: ").strip()
    X_train_s = X_train[[feat]]
    X_test_s = X_test[[feat]]

    # Train
    model = LinearRegression()
    model.fit(X_train_s, y_train)
    print(f"  Coefficient : {model.coef_[0]:.4f}")
    print(f"  Intercept   : {model.intercept_:.4f}")

    # Predict
    y_pred = model.predict(X_test_s)

    # Evaluate — Regression Metrics
    print(f"\n{'=' * 50}")
    print(f"  📊 REGRESSION RESULTS")
    print(f"{'=' * 50}")
    print(f"  MSE  : {mean_squared_error(y_test, y_pred):.4f}")
    print(f"  RMSE : {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    print(f"  MAE  : {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"  R²   : {r2_score(y_test, y_pred):.4f}")
    print(f"{'=' * 50}")

    # Visualization — Regression Line Plot
    x_vals = X_test_s.values.flatten()
    sorted_idx = np.argsort(x_vals)
    plt.figure(figsize=(8, 5))
    plt.scatter(x_vals, y_test, color='steelblue', label='Actual', alpha=0.7)
    plt.plot(x_vals[sorted_idx], y_pred[sorted_idx], color='tomato', linewidth=2, label='Regression Line')
    plt.xlabel(feat)
    plt.ylabel(target)
    plt.title('Simple Linear Regression')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run()
