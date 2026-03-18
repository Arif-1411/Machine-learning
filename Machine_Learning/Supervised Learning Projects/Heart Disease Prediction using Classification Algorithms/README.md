# Heart Disease Prediction Using Ensemble Learning

This project implements machine learning models to predict heart disease using various ensemble learning techniques. The analysis includes data exploration, preprocessing, and model building with Decision Trees, Random Forest, and AdaBoost classifiers.

## Author
Arifudheen

## Dataset
The project uses the `heart.csv` dataset containing various medical attributes for heart disease prediction.

## Project Structure

### 1. Data Exploration
- Initial data loading and inspection
- Statistical analysis using `describe()` and `info()`
- Visualization of key features:
  - Sex distribution
  - Age distribution
  - Cholesterol levels
  - Resting blood pressure
  - Maximum heart rate achieved

### 2. Data Preprocessing
- Null value detection and handling
- Target class distribution analysis
- Outlier detection using Box Plots for:
  - Age
  - Resting blood pressure (trestbps)
  - Cholesterol (chol)
  - Maximum heart rate (thalach)
  - ST depression (oldpeak)
- Outlier removal using Z-Score (threshold < 3)
  - Original dataset: 1025 samples
  - After outlier removal: 969 samples

### 3. Feature Analysis
- Correlation heatmap generation
- Feature importance visualization
- Correlation analysis between target variable and other attributes

### 4. Machine Learning Models

#### Decision Tree Classifier
- Accuracy: 100%
- Precision: 100%
- Recall: 100%
- F1-Score: 100%

#### Random Forest Classifier
- Accuracy: 100%
- Precision: 100%
- Recall: 100%
- F1-Score: 100%

#### AdaBoost Classifier
- Accuracy: 93.3%
- Precision: 94.12%
- Recall: 93.20%
- F1-Score: 93.66%

## Requirements
```python
pandas
numpy
matplotlib
seaborn
scipy
sklearn
```

## Installation
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn
```

## Usage
1. Load the dataset:
```python
df = pd.read_csv('heart.csv')
```

2. Run the preprocessing pipeline:
```python
# Remove outliers using Z-Score
z = np.abs(stats.zscore(df))
data_clean = df[(z<3).all(axis=1)]
```

3. Train and evaluate models:
```python
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train model (example with Decision Tree)
dtree = DecisionTreeClassifier(random_state=0)
dtree.fit(X_train, y_train)
y_pred = dtree.predict(X_test)
```

## Key Features
- Comprehensive exploratory data analysis (EDA)
- Multiple visualization techniques
- Statistical outlier detection and removal
- Feature correlation analysis
- Multiple ensemble learning models comparison
- Feature importance ranking
- Complete model evaluation metrics

## Model Performance Comparison
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Decision Tree | 100% | 100% | 100% | 100% |
| Random Forest | 100% | 100% | 100% | 100% |
| AdaBoost | 93.3% | 94.12% | 93.20% | 93.66% |

## Visualizations
The project includes various visualizations:
- Count plots for categorical variables
- Histograms for continuous variables
- Box plots for outlier detection
- Correlation heatmap
- Feature importance bar plots
- Line plots for feature-target correlation

## Data Split
- Training Set: 80%
- Testing Set: 20%
- Random State: 0 (for reproducibility)

## Notes
- The Decision Tree and Random Forest models show perfect accuracy on the test set, which might indicate potential overfitting
- AdaBoost shows more realistic performance metrics
- Consider implementing cross-validation for more robust model evaluation
- The dataset is relatively clean with no missing values
