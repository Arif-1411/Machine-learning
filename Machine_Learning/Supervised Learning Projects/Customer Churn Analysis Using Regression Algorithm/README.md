# 📊 Customer Churn Analysis Using Regression Algorithm

A machine learning project focused on analyzing customer churn and predicting churn probability using regression techniques and business-centric data.

---

## 📌 Project Overview

This project implements a full workflow: ingesting customer account, usage, and demographic data; conducting exploratory analysis to identify churn patterns; engineering features and applying regression models to estimate churn probability; and interpreting outcomes for business action. The objective is to quantify churn risk and provide actionable insights for retention strategies.

---

## 🧰 Tech Stack

* **Language:** Python
* **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn
* **Environment:** Jupyter Notebook / Google Colab

---

## 🔄 Workflow Summary

### 1. Data Collection

Dataset containing features like customer demographics, subscription details, usage metrics, tenure, payment behavior, and a target indicating churn (or churn probability/regression target).

### 2. Exploratory Data Analysis (EDA)

* Distribution of churn vs non-churn customers
* Visualizations of usage, tenure, payment behavior across churn status
* Correlation matrix and identifying key predictors
* Handling class imbalance or target skewness

### 3. Feature Engineering

* Encoding categorical variables (e.g., plan type, region)
* Creating derived features like tenure buckets, usage growth rate, payment lateness ratio
* Scaling/normalizing features if required by regression algorithm
* Splitting into training and validation sets

### 4. Modeling

Regression algorithms applied:

* **Linear Regression** (baseline)
* **Random Forest Regressor** or **Gradient Boosting Regressor** (better performance)
* Converting churn classification problem into probability/regression output

### 5. Evaluation

Metrics used include:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² score
* Interpretation of predicted probabilities and thresholds for decision-making

**Result:** The selected regression model provided accurate churn risk estimation and delivered business-actionable output by ranking customers by risk.

### 6. Prediction & Insights

* Generated churn-risk scores for each customer
* Identified top‐risk segments and key features influencing churn (e.g., tenure, payment lateness, usage decline)
* Recommended targeted retention strategies based on score and feature importance

---

## 📁 Project Structure

```
Customer-Churn-Analysis/
│── data/
│── notebooks/
│── src/
│── README.md
│── requirements.txt
```

---

## 📈 Key Findings

* Customers with shorter tenure, increased payment lateness and sharp drop in usage had higher churn risk
* Derived features (e.g., usage change rate, payment lateness ratio) strongly improved model performance
* Regression approach allowed ranking customers by risk rather than simple binary classification
* Business teams can now prioritise retention efforts using churn risk scores

---

## 🚀 Future Improvements

* Integrate time-series or event-sequence modelling to capture churn patterns over time
* Deploy model as interactive dashboard or API for real-time churn risk monitoring
* Include external behavioural data (e.g., customer support interactions, social sentiment) to enhance prediction
* Implement feedback loop with live data to continuously update and calibrate risk model

---

## 🧑‍💻 Author

**[Arifudheen]**
