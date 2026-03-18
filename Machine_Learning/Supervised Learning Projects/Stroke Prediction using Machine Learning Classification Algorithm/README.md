# Stroke Prediction Using Machine Learning

A comprehensive machine learning project for predicting stroke risk based on various health and lifestyle factors.

## Author
**Arifudheen**

## Dataset
**Source:** [Kaggle - Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

**File:** `healthcare-dataset-stroke-data.csv`

## Project Overview
This project implements multiple machine learning algorithms to predict the likelihood of stroke occurrence based on patient health data. The analysis includes extensive data preprocessing, handling class imbalance through oversampling, and comparison of various classification models.

---

## Dataset Description

| Feature | Description | Values |
|---------|-------------|--------|
| `id` | Unique identifier | Integer |
| `gender` | Patient's gender | Male, Female, Other |
| `age` | Patient's age | Float |
| `hypertension` | Has hypertension | 0 = No, 1 = Yes |
| `heart_disease` | Has heart disease | 0 = No, 1 = Yes |
| `ever_married` | Marital status | Yes, No |
| `work_type` | Type of work | children, Govt_job, Never_worked, Private, Self-employed |
| `Residence_type` | Residence type | Rural, Urban |
| `avg_glucose_level` | Average glucose level | Float |
| `bmi` | Body Mass Index | Float |
| `smoking_status` | Smoking status | formerly smoked, never smoked, smokes, Unknown |
| `stroke` | Target variable | 0 = No Stroke, 1 = Stroke |

---

## Project Structure

### 1. Data Loading and Exploration
- Import required libraries (pandas, numpy, seaborn, matplotlib)
- Load dataset and display initial records
- Check for missing values and data types

### 2. Data Preprocessing

#### Missing Value Treatment
- **BMI column:** 201 missing values filled with mean value

#### Feature Engineering
- Dropped `id` column (not relevant for prediction)
- Categorical variable encoding:

| Feature | Encoding |
|---------|----------|
| gender | Female=0, Male=1, Other=2 |
| ever_married | No=0, Yes=1 |
| work_type | children=0, Govt_job=1, Never_worked=2, Private=3, Self-employed=4 |
| Residence_type | Rural=0, Urban=1 |
| smoking_status | never smoked=0, formerly smoked=1, smokes=2, Unknown=3 |

### 3. Handling Class Imbalance

**Original Distribution:**
- No Stroke (0): 4,861 samples
- Stroke (1): 249 samples

**After Oversampling (using sklearn's resample):**
- No Stroke (0): 4,861 samples
- Stroke (1): 4,861 samples (upsampled)

### 4. Data Correlation Analysis
- Correlation heatmap generated to understand feature relationships

---

## Machine Learning Models

### Train-Test Split
- **Training Set:** 80%
- **Testing Set:** 20%
- **Random State:** 7

### Model Performance Comparison

| Model | Accuracy |
|-------|----------|
| **Random Forest** | **98.92%** |
| Decision Tree | 96.25% |
| XGBoost | 83.60% |
| Logistic Regression | 77.48% |

---

## Detailed Model Results

### 1. Random Forest Classifier (Best Model)

**Classification Report:**
```
              precision    recall  f1-score   support

           0       1.00      0.98      0.99       988
           1       0.98      1.00      0.99       957

    accuracy                           0.99      1945
```

**Feature Importance Ranking:**
| Rank | Feature | Importance |
|------|---------|------------|
| 1 | age | 0.398422 |
| 2 | avg_glucose_level | 0.199334 |
| 3 | bmi | 0.176850 |
| 4 | smoking_status | 0.051561 |
| 5 | work_type | 0.046476 |
| 6 | ever_married | 0.035515 |
| 7 | hypertension | 0.028047 |
| 8 | gender | 0.022215 |
| 9 | Residence_type | 0.021167 |
| 10 | heart_disease | 0.020412 |

**ROC-AUC Score:** ~0.99

### 2. Decision Tree Classifier
- **Accuracy:** 96.25%

### 3. XGBoost Classifier

**Classification Report:**
```
              precision    recall  f1-score   support

           0       0.89      0.77      0.83       988
           1       0.79      0.90      0.84       957

    accuracy                           0.84      1945
```

**Additional Analysis:**
- Early stopping implemented (10 rounds)
- Log loss monitoring during training
- Decision tree visualization available

### 4. Logistic Regression

**Classification Report:**
```
              precision    recall  f1-score   support

           0       0.80      0.74      0.77       988
           1       0.75      0.82      0.78       957

    accuracy                           0.77      1945
```

**ROC-AUC Score:** ~0.85

**Note:** Convergence warning observed - may require increased iterations or data scaling

---

## Visualizations Included

1. **Class Distribution:** Count plot of stroke vs non-stroke cases
2. **Correlation Heatmap:** Feature correlation analysis
3. **Confusion Matrices:** For all models
4. **ROC Curves:** For XGBoost, Logistic Regression, and Random Forest
5. **Log Loss Plot:** XGBoost training progress
6. **Decision Tree Visualization:** XGBoost tree structure
7. **Probability Histograms:** Predicted probability distributions
8. **Feature Importance Bar Chart:** Random Forest feature rankings

---

## Interactive Prediction

The project includes an interactive prediction system where users can input their health parameters:

### Input Parameters:
```
- Gender (0 = Female, 1 = Male)
- Age
- Hypertension (0 = No, 1 = Yes)
- Heart Disease (0 = No, 1 = Yes)
- Ever Married (0 = No, 1 = Yes)
- Work Type (0-4 scale)
- Residence Type (0 = Rural, 1 = Urban)
- Average Glucose Level
- BMI
- Smoking Status (0-3 scale)
```

### Output:
- Probability of not having stroke
- Probability of having stroke

### Example Prediction:
```python
# Input: Male, 21 years, no hypertension, no heart disease, 
#        not married, self-employed, urban, glucose=109, BMI=15, never smoked

# Output:
# Prob of dont have stroke: 0.99
# Prob of have stroke: 0.01
```

---

## Requirements

```python
pandas
numpy
seaborn
matplotlib
scikit-learn
xgboost
```

## Installation

```bash
pip install pandas numpy seaborn matplotlib scikit-learn xgboost
```

---

## Usage

### 1. Load and Preprocess Data
```python
import pandas as pd

df = pd.read_csv('healthcare-dataset-stroke-data.csv')
df['bmi'].fillna(int(df['bmi'].mean()), inplace=True)
```

### 2. Handle Class Imbalance
```python
from sklearn.utils import resample

df_majority = df2[(df2['stroke']==0)] 
df_minority = df2[(df2['stroke']==1)] 

df_minority_upsampled = resample(df_minority, 
                                 replace=True,
                                 n_samples=4861,
                                 random_state=42)

df_upsampled = pd.concat([df_minority_upsampled, df_majority])
```

### 3. Train Random Forest Model
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
```

### 4. Make Predictions
```python
# Single prediction
Xnew = [[1, 70, 1, 1, 0, 3, 1, 100, 40, 2]]
probability = rfc.predict_proba(Xnew)
```

---

## Key Findings

1. **Age is the most important predictor** of stroke risk (39.8% importance)
2. **Average glucose level** and **BMI** are the next most significant factors
3. **Random Forest** outperforms other models with 98.92% accuracy
4. **Class imbalance** was successfully addressed using oversampling
5. **Lifestyle factors** (smoking, work type) contribute moderately to predictions

---

## Model Selection Recommendation

**Recommended Model: Random Forest Classifier**

**Reasons:**
- Highest accuracy (98.92%)
- Excellent precision and recall for both classes
- ROC-AUC score near 1.0
- Provides interpretable feature importance
- Robust to overfitting with default parameters

---

## Limitations and Future Work

### Limitations:
- Oversampling may introduce some bias
- High accuracy might indicate slight overfitting
- Logistic Regression convergence issues

### Future Improvements:
- Implement cross-validation for more robust evaluation
- Try SMOTE for synthetic minority oversampling
- Feature selection based on importance scores
- Hyperparameter tuning using GridSearchCV
- Deploy model as web application

---

## File Structure

```
stroke-prediction/
│
├── healthcare-dataset-stroke-data.csv    # Dataset
├── stroke_prediction.ipynb               # Main notebook
├── README.md                             # Documentation
```

---

