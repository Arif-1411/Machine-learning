# 💵 Banknote Authentication — ML Classification Project

Forged or genuine? This project uses **machine learning** to detect counterfeit banknotes based on wavelet-transformed image features extracted from banknote photographs.

---

## 📌 Project Overview

Banks and financial institutions lose billions every year to counterfeit currency. This project trains and compares multiple classification models — **Decision Tree, Random Forest, XGBoost, and Logistic Regression** — on wavelet-transform features of banknote images to distinguish genuine notes from forgeries.

**Best Model:** XGBoost & Random Forest — both achieved **~99.31% accuracy**

---

## 🗂️ Dataset

- **Source:** [Banknote Authentication — UCI ML Repository via Kaggle](https://www.kaggle.com/datasets/shantanuss/banknote-authentication-uci)
- **Records:** 1,372 samples → cleaned to 1,210 → balanced to 1,454 (after upsampling)
- **Features:**

| Feature | Description |
|---------|-------------|
| `variance` | Variance of Wavelet Transformed image |
| `skewness` | Skewness of Wavelet Transformed image |
| `curtosis` | Kurtosis of Wavelet Transformed image |
| `entropy` | Entropy of Wavelet Transformed image |
| `class` | **Target** — 0 = Genuine, 1 = Forged |

---

## 🔄 Workflow

```
Raw Data
   │
   ▼
Exploratory Data Analysis (Boxplots per feature)
   │
   ▼
Outlier Removal (Z-score threshold = 2) → 1,210 samples
   │
   ▼
Class Balancing (Upsampling minority class) → 727 + 727 = 1,454 samples
   │
   ▼
Correlation Heatmap
   │
   ▼
Train-Test Split (80/20)
   │
   ▼
Model Training & Evaluation
   │
   ▼
Best Model: XGBoost / Random Forest (~99.31%)
```

---

## 🤖 Models & Results

| Model | Accuracy |
|-------|----------|
| Logistic Regression | 98.97% |
| Decision Tree | 98.97% |
| **Random Forest** | **99.31%** |
| **XGBoost** | **99.31%** |

---

## 📊 Visualizations Included

- 📦 **Boxplots** — Outlier detection for all 4 features
- 🔢 **Countplot** — Class distribution before & after balancing
- 🔥 **Correlation Heatmap** — Feature relationships
- 🧩 **Confusion Matrix** — XGBoost prediction breakdown
- 📈 **ROC Curve** — AUC score visualization (AUC ≈ 0.9997)
- 🌲 **Feature Importance Plot** — XGBoost feature ranking
- 🌳 **Decision Tree Visualization** — XGBoost tree structure

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Core language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib & Seaborn | Visualization |
| Scikit-learn | ML models + preprocessing |
| XGBoost | Gradient boosting classifier |
| SciPy | Z-score outlier removal |

---

## ▶️ How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/Arif-1411/<repo-name>.git
cd <repo-name>
```

### 2. Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost scipy
```

### 3. Download Dataset
Download from Kaggle:
👉 [Banknote Authentication UCI](https://www.kaggle.com/datasets/shantanuss/banknote-authentication-uci)

Place the file as:
```
BankNote_Authentication.csv
```
in the same folder as the notebook.

### 4. Open & Run the Notebook
```bash
jupyter notebook
```
Open the `.ipynb` file and run all cells top to bottom.

---

## 📁 Project Structure

```
📦 banknote-authentication/
 ┣ 📓 banknote_authentication.ipynb   ← Main notebook
 ┣ 📄 BankNote_Authentication.csv     ← Dataset (download separately)
 ┗ 📄 README.md                       ← You are here
```

---

## 🔑 Key Insights

- **Variance** is the most important feature for classification (as shown by XGBoost feature importance)
- The dataset had a **class imbalance** (727 genuine vs 483 forged after cleaning) — fixed using upsampling
- Outlier removal using **Z-score (threshold = 2)** improved model quality
- All models performed exceptionally well — wavelet features are highly discriminative for this task

---

## 👤 Author

**Arifudheen T**
Data Science Engineer | Nexora Technologies
🔗 [GitHub: Arif-1411](https://github.com/Arif-1411)

---

