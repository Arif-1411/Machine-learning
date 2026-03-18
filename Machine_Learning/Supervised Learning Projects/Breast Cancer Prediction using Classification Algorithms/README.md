# Breast Cancer Prediction Using Machine Learning

## 📋 Project Overview

This project implements multiple machine learning algorithms to predict whether a breast tumor is **malignant (cancerous)** or **benign (non-cancerous)** based on features computed from digitized images of cell nuclei.

## 🎯 Problem Statement

Breast cancer is the most common cancer affecting women and accounts for nearly 1 in 3 cancers diagnosed among women in the United States. Early and accurate detection is crucial for effective treatment. This project aims to classify tumors using various ML techniques to achieve high prediction accuracy.

## 📊 Dataset

- **Source:** UCI Machine Learning Repository (Wisconsin Breast Cancer Dataset)
- **Samples:** 569 (357 Benign, 212 Malignant)
- **Features:** 30 real-value features computed from cell nuclei images
- **Target:** Binary classification (0 = Malignant, 1 = Benign)

### Features Include:
- Mean radius, texture, perimeter, area, smoothness
- Mean compactness, concavity, concave points, symmetry
- Fractal dimension (mean, standard error, and worst values)

## 🛠️ Technologies Used

| Category | Tools/Libraries |
|----------|----------------|
| Language | Python 3.9+ |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost |
| Preprocessing | StandardScaler, PCA |

## 📈 Models Implemented

| Model | Test Accuracy | CV Score |
|-------|---------------|----------|
| Logistic Regression | 96.5% | 98.6% |
| Decision Tree | 90.2% | 94.6% |
| Random Forest | 93.7% | 95.8% |
| K-Nearest Neighbors | 95.1% | 97.4% |
| Gaussian Naive Bayes | 91.6% | 92.5% |
| Support Vector Classifier | **97.2%** | **99.1%** |
| XGBoost | 95.8% | 96.9% |
| **Stacked Ensemble** | **95.8%** | **100%** |

## 🔄 Pipeline Architecture

```
Data → StandardScaler → PCA (dimensionality reduction) → ML Model → Prediction
```

## 📁 Project Structure

```
breast-cancer-prediction/
│
├── breast_cancer_prediction.ipynb    # Main Jupyter notebook
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
└── results/                           # Confusion matrices & plots
```

## 🚀 Quick Start

### Prerequisites
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

### Run the Project
```bash
jupyter notebook breast_cancer_prediction.ipynb
```

## 📊 Key Results

### Best Performing Model: Support Vector Classifier (SVC)
- **Test Accuracy:** 97.2%
- **Parameters:** `C=100, gamma=0.001, kernel='rbf', n_components=8`

### Classification Report (SVC)
```
              precision    recall  f1-score   support

   Malignant       0.98      0.94      0.96        53
      Benign       0.97      0.99      0.98        90

    accuracy                           0.97       143
```

### Confusion Matrix Interpretation
- **True Negatives (Malignant correctly identified):** 50
- **True Positives (Benign correctly identified):** 89
- **False Positives:** 3
- **False Negatives:** 1

## 🔬 Methodology

### 1. Data Preprocessing
- ✅ No missing values detected
- ✅ No duplicate records found
- ✅ Feature scaling using StandardScaler
- ✅ Stratified train-test split (75-25)

### 2. Dimensionality Reduction
- PCA applied to reduce features while preserving variance
- Optimal components selected via GridSearchCV

### 3. Model Tuning
- GridSearchCV with 5-fold cross-validation
- Hyperparameter optimization for each model

### 4. Ensemble Learning
- Stacking classifier combining all 7 models
- Meta-learner: SVC with RBF kernel

## 📉 Visualizations Included

- Class distribution (Benign vs Malignant)
- PCA component scatter plot
- Confusion matrices for all models
- Correlation heatmap

## 🎓 Key Insights

1. **SVC outperforms** other individual models with 97.2% accuracy
2. **PCA reduces** dimensionality effectively without significant information loss
3. **Stacking ensemble** achieves perfect CV score but similar test accuracy
4. **Feature scaling** is critical for distance-based algorithms (KNN, SVC)

## 📝 Future Improvements

- [ ] Implement deep learning models (Neural Networks)
- [ ] Add SHAP values for model interpretability
- [ ] Deploy as web application using Flask/Streamlit
- [ ] Cross-validate on external datasets

## 👥 Author

**[Arifudheen]**

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the dataset
- Scikit-learn documentation and community

---

**⭐ Final Accuracy Achieved: 97%**