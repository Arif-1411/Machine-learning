# 💎 Diamond Price Prediction using Regression Algorithms

A machine-learning project focused on estimating the price of diamonds using regression models, exploring how features like carat, cut, colour, clarity, depth, and table influence price.

---

### Dataset Source 

* [Kaggle](https://www.kaggle.com/datasets/ronil8/diamond-price-prediction-dataset)

## 📌 Project Overview

This project constructs a full pipeline: gathering diamond dataset, exploring relationships between features and price, engineering features, training regression models, and evaluating their predictive accuracy. The goal is to build a model that can estimate diamond prices accurately and reveal which features matter most.

---

## 🧰 Tech Stack

* **Language:** Python
* **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn
* **Environment:** Jupyter Notebook / Google Colab

---

## 🔄 Workflow Summary

### 1. Data Collection

Dataset includes diamond attributes such as carat, cut, color, clarity, depth, table, and price (target). Often derived from public datasets (e.g., “diamonds” from R’s ggplot2).

### 2. Exploratory Data Analysis (EDA)

* Visualisations: carat vs price, price distribution, boxplots of price by cut/color/clarity
* Correlation matrix amongst numeric features & price
* Identified skew in price/carats and possibly log-transformed target for modelling

---

### 3. Feature Engineering

* Encoded categorical features (cut, color, clarity) using one-hot encoding or ordinal mapping
* Derived features such as carat squared, carat–depth interaction, price per carat
* Log-transformation of price and/or carat if skew-distribution present
* Split data into train/test sets

---

### 4. Modelling

Regression algorithms utilised:

* **Linear Regression** (baseline)
* **Random Forest Regressor** (strong performance)
* **Gradient Boosting Regressor** or **XGBoost** for further improvement
* Hyper-parameter tuning via cross-validation on e.g., n estimators, max_depth, learning_rate

---

### 5. Evaluation

Metrics used to assess model performance:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² (coefficient of determination)
  **Result:** The best performing model achieved low RMSE and high R², indicating good predictive capability and feature importance insights.

---

### 6. Prediction & Insights

* Generated predictions for individual diamonds and compared predicted vs actual prices
* Analysed feature importances: carat emerged as strongest predictor; cut, clarity and colour also influenced price
* Provided business-relevant insight: e.g., premium paid for higher clarity or better cut after controlling for carat

---

## 📁 Project Structure

```
Diamond-Price-Prediction/
│── data/
│── notebooks/
│── src/
│── README.md
│── requirements.txt
```

---

## 📈 Key Findings

* Carat is the dominant predictor of price, but quality factors (cut, clarity, colour) also add substantial value.
* Log-transforming the target improved model residual distribution and model fit.
* Tree-based regressors outperformed linear regression due to non-linear relationships between features and price.
* Derived features such as carat–depth interaction boosted model accuracy.

---

## 🚀 Future Improvements

* Include external or market data (e.g., diamond dealer premiums, regional pricing) for richer features.
* Deploy model via web app for users (e.g., clients estimating their diamond value).
* Use ensemble stacking or model blending to further reduce prediction error.
* Incorporate explainability (e.g., SHAP values) to interpret individual predictions for users.
* Monitor model reliability over time as diamond market dynamics may shift.

---

## 🧑‍💻 Author

**[Arifudheen]**

---
