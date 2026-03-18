# 🛡️ Insurance Premium Prediction using Regression Algorithms

A machine learning project focused on estimating insurance premium costs using regression models, insured demographics, policy attributes, and historical claims data.

---

## 📌 Project Overview

This project covers the complete workflow: ingesting data on insured individuals (age, sex, BMI, smoking status, region, prior claims), performing exploratory data analysis to uncover cost determinants, engineering new features, building regression models, and evaluating their performance. The objective is to provide accurate premium estimations and insights into cost-drivers.

---

## 🧰 Tech Stack

* **Language:** Python
* **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn
* **Environment:** Jupyter Notebook / Google Colab

---

## 🔄 Workflow Summary

### 1. Data Collection

Dataset includes variables like: age, sex, BMI, smoking status, children, region, prior insurance claims, policy type, and a target variable for annual premium cost.

### 2. Exploratory Data Analysis (EDA)

* Distribution of premium cost values across demographics
* Visualizations of premium vs features such as age, BMI, smoking status
* Correlation matrix to identify key relationships
* Detection of missing values, outliers and skewed target distribution

### 3. Feature Engineering

* Encoding categorical variables (e.g., sex, smoking status, region)
* Creating derived features like BMI category, smoker × age interaction, number of dependents bucket
* Log-transforming skewed target or heavy-tailed features
* Scaling numerical features for regression compatibility
* Splitting dataset into training and test sets

### 4. Modeling

Regression algorithms applied include:

* **Linear Regression** (baseline)
* **Random Forest Regressor** or **Gradient Boosting Regressor** (strong performers)
* **(Optional) XGBoost / LightGBM** for improved estimation

### 5. Evaluation

Metrics used to assess model performance:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² score
* Residuals analysis and distribution

**Result:** The top performing regression model achieved low error rates and high R², demonstrating reliable premium estimates and highlighting key cost-drivers such as smoking status, BMI and age.

### 6. Prediction & Insights

* Generated premium estimates for new policyholder profiles
* Analysed feature importance: age, smoking status, BMI category emerged as leading predictors
* Provided actionable insights: smokers and higher BMI categories significantly raise premium cost; demographic segmentation for risk-based pricing

---

## 📁 Project Structure

```
Insurance-Premium-Prediction/
│── data/
│── notebooks/
│── src/
│── README.md
│── requirements.txt
```

---

## 📈 Key Findings

* Smoker status had the most significant impact on premium cost, followed by BMI and age
* Interaction features and derived feature transformations improved estimation accuracy
* Regression models such as Random Forest and Gradient Boosting outperformed linear baseline
* The prediction framework supports insurers with data-driven premium pricing and risk assessment

---

## 🚀 Future Improvements

* Include additional data such as claims history, policy duration, and external health indicators (e.g., fitness tracker data)
* Explore deep learning and sequence models (e.g., for recurring claims or longitudinal insured behavior)
* Deploy as a web or mobile app (Flask/Streamlit) for real-time premium prediction and quoting
* Implement fairness and bias analysis to ensure equitable pricing across demographics
* Enable continuous model update with real-world feedback and claim-outcome data

---

## 🧑‍💻 Author

**[Arifudheen]**
