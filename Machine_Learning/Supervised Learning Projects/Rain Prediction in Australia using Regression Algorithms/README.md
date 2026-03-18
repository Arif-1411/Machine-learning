# 🌦️ Rain Prediction in Australia Using Regression Algorithms

A machine-learning project focused on estimating the amount of rainfall (or next-day rainfall value) in Australia using regression models and meteorological data.

---

## 📌 Project Overview

This project builds an end-to-end regression pipeline: ingesting historical weather data for Australia, performing exploratory analysis, engineering features (temperature, humidity, wind, pressure, previous rainfall, etc.), training regression models, evaluating their performance, and deriving actionable insights. The goal is to predict the **rainfall amount** for the next day (or another target) and understand which features contribute most.

---

## 🧰 Tech Stack

* **Language:** Python
* **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn
* **Environment:** Jupyter Notebook / Google Colab

---

## 🔄 Workflow Summary

### 1. Data Collection

Historical weather observations for Australian locations: features such as MinTemp, MaxTemp, Rainfall, Evaporation, Sunshine, WindGustSpeed, Humidity3pm, Pressure9am, etc., with target variable being the next day’s rainfall (or rainfall amount).

### 2. Exploratory Data Analysis (EDA)

* Visualise rainfall distribution (skew, outliers)
* Scatterplots / boxplots of rainfall vs key predictors (e.g., Humidity3pm, RainToday)
* Correlation matrix among numeric features and target
* Handle missing values, outlier detection/handling, target skew‐transformation

### 3. Feature Engineering

* Encode categorical features (Location, Wind direction, etc.) via one-hot or ordinal encoding
* Derive features: e.g., previous day rainfall flag, difference between min and max temperature, ratio of humidity to pressure
* Log-transform target or large skew numeric predictors if needed
* Split into training and test sets (e.g., 80/20) and apply scaling if required

### 4. Modelling

Regression algorithms used:

* **Linear Regression** (baseline)
* **Random Forest Regressor** and/or **Gradient Boosting Regressor** for improved non-linear modelling
* Possibly tuned hyper-parameters (max_depth, n_estimators, learning_rate) via cross-validation

### 5. Evaluation

Metrics used to assess performance:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² (coefficient of determination)
  **Result:** Best model achieved low error on the test set, showing good capability to predict rainfall amounts.

### 6. Prediction & Insights

* Generated predictions for new weather inputs and compared actual vs predicted rainfall
* Analysed feature importance: e.g., Humidity3pm, RainToday flag, WindGustSpeed, Pressure9am emerged as key predictors
* Provided insight for use cases: farmers, planners or weather-aware services could use predictions for irrigation scheduling, event planning, or risk mitigation

---

## 📁 Project Structure

```
Rain-Prediction-Australia-Regression/
│── data/
│   ├── raw/
│   └── processed/
│── notebooks/
│   └── rain_prediction_regression.ipynb
│── src/
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── models.py
│   └── evaluate.py
│── README.md
│── requirements.txt
```

---

## 📈 Key Findings

* Regression modelling is viable for estimating rainfall amounts when rich meteorological features are available.
* Features like **humidity at 3pm**, **rainfall today**, and **pressure at 9am** consistently ranked among the most important predictors.
* Skew in target (many days with zero or low rainfall) required transformation or special handling.
* Non-linear models (tree-based regressors) outperformed linear models due to complex interactions among weather features.

---

## 🚀 Future Improvements

* Extend to **time-series modelling**: using past n days of data (e.g., 7-day window) to predict rainfall, deploy LSTM/GRU or temporal regression models.
* Incorporate external data sources: satellite imagery, radar reflectivity, land elevation, micro-climate sensors for richer features.
* Deploy as a web service or dashboard: users input current weather metrics and receive next‐day rainfall estimate.
* Add interpretability: SHAP or LIME to explain individual predictions and build user trust.
* Retrain periodically to adapt to climate trends or changes in data distribution (concept drift).

---

## 🧑‍💻 Author

**[Arifudheen]**

---
