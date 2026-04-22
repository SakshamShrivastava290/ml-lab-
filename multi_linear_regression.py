"""
=============================================================
  MULTI-LINEAR REGRESSION — Standalone Lab Program
=============================================================
  Covers: Multiple Linear Regression (all features)
  Metrics: MSE, MAE, R² Score
  Visualization: Actual vs Predicted scatter plot
  Libraries: pandas, numpy, matplotlib, sklearn
  Usage: python multi_linear_regression.py
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


def run():
    print("\n" + "=" * 60)
    print("  📈 MULTI-LINEAR REGRESSION")
    print("=" * 60)

    path = input("Enter CSV path (default: data.csv): ").strip() or "data.csv"
    target = input("Enter target column: ").strip()

    X_train, X_test, y_train, y_test = load_and_preprocess(path, target)

    # Train with all features
    model = LinearRegression()
    model.fit(X_train, y_train)
    print(f"  Coefficients : {np.round(model.coef_, 4)}")
    print(f"  Intercept    : {model.intercept_:.4f}")

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate — Regression Metrics
    print(f"\n{'=' * 50}")
    print(f"  📊 REGRESSION RESULTS")
    print(f"{'=' * 50}")
    print(f"  MSE  : {mean_squared_error(y_test, y_pred):.4f}")
    print(f"  RMSE : {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    print(f"  MAE  : {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"  R²   : {r2_score(y_test, y_pred):.4f}")
    print(f"{'=' * 50}")

    # Visualization — Actual vs Predicted
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, color='steelblue', edgecolors='k', alpha=0.7)
    mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], color='tomato', linewidth=2, linestyle='--', label='Ideal Fit')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Multi-Linear Regression — Actual vs Predicted')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run()
