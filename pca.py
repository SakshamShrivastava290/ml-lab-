"""
=============================================================
  PCA (Principal Component Analysis) — Standalone Lab Program
=============================================================
  Metrics: Explained Variance Ratio
  Visualization: Scree Plot (individual + cumulative)
  Libraries: pandas, numpy, matplotlib, sklearn
  Usage: python pca.py
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA


def load_and_preprocess_unsupervised(path, drop_col=None):
    """Load and preprocess for unsupervised learning (no target)."""
    df = pd.read_csv(path)
    print(f"✅ Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

    if drop_col and drop_col in df.columns:
        df = df.drop(columns=[drop_col])
        print(f"  Dropped column: '{drop_col}'")

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

    features = list(df.columns)
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(df), columns=features)
    print(f"  Scaled {len(features)} features\n")
    return X, features


def run():
    print("\n" + "=" * 60)
    print("  📉 PRINCIPAL COMPONENT ANALYSIS (PCA)")
    print("=" * 60)

    path = input("Enter CSV path (default: data.csv): ").strip() or "data.csv"
    drop = input("Drop any column? (leave blank to skip): ").strip()

    X, features = load_and_preprocess_unsupervised(path, drop if drop else None)

    max_comp = min(X.shape[0], X.shape[1])
    n = input(f"Number of components (max {max_comp}, default all): ").strip()
    n_components = int(n) if n else max_comp

    # Fit PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    evr = pca.explained_variance_ratio_
    cumulative = np.cumsum(evr)

    # Print Results
    print(f"\n{'=' * 50}")
    print(f"  📉 PCA Results")
    print(f"{'=' * 50}")
    for i, (var, cum) in enumerate(zip(evr, cumulative)):
        print(f"  PC{i+1}: Variance = {var:.4f}  |  Cumulative = {cum:.4f}")
    print(f"{'=' * 50}")
    print(f"  Total explained: {cumulative[-1]:.4f}")
    print(f"{'=' * 50}")

    # Visualization — Scree Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Individual variance bar chart
    axes[0].bar(range(1, len(evr)+1), evr, color='steelblue', edgecolor='black')
    axes[0].set_xlabel('Principal Component')
    axes[0].set_ylabel('Explained Variance Ratio')
    axes[0].set_title('Scree Plot — Individual Variance')
    axes[0].set_xticks(range(1, len(evr)+1))

    # Cumulative variance line plot
    axes[1].plot(range(1, len(cumulative)+1), cumulative,
                 marker='o', color='tomato', linewidth=2)
    axes[1].axhline(y=0.95, color='gray', linestyle='--', label='95% threshold')
    axes[1].set_xlabel('Number of Components')
    axes[1].set_ylabel('Cumulative Explained Variance')
    axes[1].set_title('Cumulative Explained Variance')
    axes[1].set_xticks(range(1, len(cumulative)+1))
    axes[1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run()
