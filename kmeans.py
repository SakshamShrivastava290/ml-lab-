"""
=============================================================
  K-MEANS CLUSTERING — Standalone Lab Program
=============================================================
  Metrics: Inertia (WCSS), Silhouette Score
  Visualization: 2D Cluster Scatter Plot
  Libraries: pandas, numpy, matplotlib, sklearn
  Usage: python kmeans.py
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


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
    print("  🔵 K-MEANS CLUSTERING")
    print("=" * 60)

    path = input("Enter CSV path (default: data.csv): ").strip() or "data.csv"
    drop = input("Drop any column? (leave blank to skip): ").strip()

    X, features = load_and_preprocess_unsupervised(path, drop if drop else None)

    k = input("Enter number of clusters K (default 3): ").strip()
    k = int(k) if k else 3

    # Train K-Means
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X)

    labels = model.labels_
    inertia = model.inertia_

    # Evaluate — Clustering Metrics
    print(f"\n{'=' * 50}")
    print(f"  🔵 K-Means Clustering Results")
    print(f"{'=' * 50}")
    print(f"  Clusters         : {k}")
    print(f"  Inertia (WCSS)   : {inertia:.4f}")
    if k > 1:
        sil = silhouette_score(X, labels)
        print(f"  Silhouette Score : {sil:.4f}")
    print(f"{'=' * 50}")

    # Visualization — 2D Cluster Plot (first two features)
    X_arr = np.array(X)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_arr[:, 0], X_arr[:, 1], c=labels, cmap='viridis',
                          edgecolors='k', s=60, alpha=0.8)
    centers = model.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X',
                s=200, edgecolors='k', linewidths=2, label='Centroids')
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.title(f'K-Means Clustering (K={k})')
    plt.colorbar(scatter, label='Cluster')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run()
