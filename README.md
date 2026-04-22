# 📘 ML Lab Programs — Quick Reference

All 12 programs are **standalone files**. Each one has its own preprocessing, training, evaluation, and visualization built in. Just run the file you need with **any CSV dataset**.

---

## 📦 Setup (one-time)

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## 📂 Files Overview

| # | File | What It Does | Problem Type |
|---|------|-------------|:------------|
| 1 | `pandas_io.py` | Load, explore, export data | Data Handling |
| 2 | `preprocessing.py` | Missing values, encoding, scaling, split | Data Prep |
| 3 | `linear_regression.py` | Simple Linear Regression (1 feature) | Regression |
| 4 | `multi_linear_regression.py` | Multi-Linear Regression (all features) | Regression |
| 5 | `decision_tree.py` | Decision Tree (ID3 entropy) | Classification / Regression |
| 6 | `logistic_regression.py` | Logistic Regression | Classification |
| 7 | `knn.py` | K-Nearest Neighbors | Classification / Regression |
| 8 | `svm.py` | Support Vector Machine (RBF kernel) | Classification / Regression |
| 9 | `random_forest.py` | Random Forest | Classification / Regression |
| 10 | `boosting.py` | AdaBoost | Classification / Regression |
| 11 | `kmeans.py` | K-Means Clustering | Clustering |
| 12 | `pca.py` | Principal Component Analysis | Dim. Reduction |

---

## 🚀 How to Run Each File

### 1. Pandas Import/Export
```bash
python pandas_io.py
```
**Prompts:** CSV path → Displays shape, dtypes, head, tail, describe, info, missing values → Optional export to CSV/Excel

**Example interaction:**
```
Enter CSV file path (default: data.csv): data.csv
Export data? (csv / excel / skip): skip
```

---

### 2. Data Preprocessing
```bash
python preprocessing.py
```
**Prompts:** CSV path → Shows each preprocessing step → Enter target column

**Example interaction:**
```
Enter CSV file path (default: data.csv): data.csv
Enter target column name: Purchased
```
**Output:** Missing value handling → Label encoding → Standard scaling → Train-test split

---

### 3. Simple Linear Regression
```bash
python linear_regression.py
```
**Prompts:** CSV path → Target column → Pick one feature for regression

**Example interaction:**
```
Enter CSV path (default: data.csv): data.csv
Enter target column: Income
Enter feature for simple regression: Age
```
**Output:** Coefficient, Intercept, MSE, MAE, R² Score + Regression line plot

---

### 4. Multi-Linear Regression
```bash
python multi_linear_regression.py
```
**Prompts:** CSV path → Target column (uses all remaining columns as features)

**Example interaction:**
```
Enter CSV path (default: data.csv): data.csv
Enter target column: Income
```
**Output:** Coefficients, MSE, MAE, R² Score + Actual vs Predicted scatter plot

---

### 5. Decision Tree (ID3)
```bash
python decision_tree.py
```
**Prompts:** CSV path → Target column → **Classification or Regression? (c/r)**

**Example interaction:**
```
Enter CSV path (default: data.csv): data.csv
Enter target column: Purchased
Classification or Regression? (c/r): c
```
**Output (classification):** Accuracy, Precision, Recall, F1 Score + Confusion Matrix heatmap
**Output (regression):** MSE, MAE, R² Score + Actual vs Predicted plot

---

### 6. Logistic Regression
```bash
python logistic_regression.py
```
**Prompts:** CSV path → Target column

**Example interaction:**
```
Enter CSV path (default: data.csv): data.csv
Enter target column: Purchased
```
**Output:** Accuracy, Precision, Recall, F1 Score + Confusion Matrix heatmap

---

### 7. KNN
```bash
python knn.py
```
**Prompts:** CSV path → Target column → **Classification or Regression? (c/r)** → K

**Example interaction:**
```
Enter CSV path (default: data.csv): data.csv
Enter target column: Purchased
Classification or Regression? (c/r): c
Enter K (number of neighbors, default 5): 5
```
**Output (classification):** Accuracy, Precision, Recall, F1 Score + Confusion Matrix heatmap
**Output (regression):** MSE, MAE, R² Score + Actual vs Predicted plot

---

### 8. SVM
```bash
python svm.py
```
**Prompts:** CSV path → Target column → **Classification or Regression? (c/r)**

**Example interaction:**
```
Enter CSV path (default: data.csv): data.csv
Enter target column: Income
Classification or Regression? (c/r): r
```
**Output (classification):** Accuracy, Precision, Recall, F1 Score + Confusion Matrix heatmap
**Output (regression):** MSE, MAE, R² Score + Actual vs Predicted plot

---

### 9. Random Forest
```bash
python random_forest.py
```
**Prompts:** CSV path → Target column → **Classification or Regression? (c/r)** → Number of trees

**Example interaction:**
```
Enter CSV path (default: data.csv): data.csv
Enter target column: Purchased
Classification or Regression? (c/r): c
Number of trees (default 100): 100
```
**Output (classification):** Accuracy, Precision, Recall, F1 Score + Confusion Matrix heatmap
**Output (regression):** MSE, MAE, R² Score + Actual vs Predicted plot

---

### 10. Boosting (AdaBoost)
```bash
python boosting.py
```
**Prompts:** CSV path → Target column → **Classification or Regression? (c/r)** → Number of estimators

**Example interaction:**
```
Enter CSV path (default: data.csv): data.csv
Enter target column: Purchased
Classification or Regression? (c/r): c
Number of estimators (default 50): 50
```
**Output (classification):** Accuracy, Precision, Recall, F1 Score + Confusion Matrix heatmap
**Output (regression):** MSE, MAE, R² Score + Actual vs Predicted plot

---

### 11. K-Means Clustering
```bash
python kmeans.py
```
**Prompts:** CSV path → Column to drop (optional) → Number of clusters K

**Example interaction:**
```
Enter CSV path (default: data.csv): data.csv
Drop any column? (leave blank to skip): Purchased
Enter number of clusters K (default 3): 3
```
**Output:** Inertia (WCSS), Silhouette Score + 2D Cluster scatter plot with centroids

---

### 12. PCA
```bash
python pca.py
```
**Prompts:** CSV path → Column to drop (optional) → Number of components

**Example interaction:**
```
Enter CSV path (default: data.csv): data.csv
Drop any column? (leave blank to skip): Purchased
Number of components (max 6, default all):
```
**Output:** Explained variance ratio per component + Scree plot (bar + cumulative line)

---

## 🔑 Tips for Lab Exam

1. **Any CSV works** — every file asks for the path and target column at runtime
2. **Default is `data.csv`** — just press Enter if your file is named `data.csv`
3. **Preprocessing is automatic** — missing values, encoding, and scaling are handled inside every file
4. **Correct metrics per problem type:**
   - Classification → Accuracy, Precision, Recall, F1, Confusion Matrix
   - Regression → MSE, MAE, R²
   - Clustering → Inertia, Silhouette Score
   - PCA → Explained Variance Ratio

5. **For clustering/PCA** — drop the target column when prompted (these are unsupervised)
6. **Close plot windows** to continue program execution
