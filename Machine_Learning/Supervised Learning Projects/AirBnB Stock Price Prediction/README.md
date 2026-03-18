# 📈 Airbnb Stock Price Prediction

A machine learning project that predicts Airbnb (ABNB) stock closing prices using historical stock market data. The goal is to develop an effective forecasting system to assist investors and analysts in decision-making.

---

## 📌 Project Overview

This project applies supervised machine learning techniques — specifically **Decision Tree Regressor** and **Random Forest Regressor** — to predict Airbnb's daily closing stock prices based on OHLCV (Open, High, Low, Close, Volume) data along with engineered date features.

---

## 📂 Dataset


- **Features:**
  - `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`
  - Engineered: `Year`, `Month`, `DayOfWeek`

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core programming language |
| Pandas | Data loading and preprocessing |
| NumPy | Numerical computations |
| Matplotlib & Seaborn | Data visualization |
| Scikit-learn | Machine learning models and evaluation |

---

## ⚙️ Project Workflow

### 1. Data Loading & Preprocessing
- Loaded CSV data from a remote URL
- Converted `Date` column to `datetime` format and set it as the index

### 2. Feature Engineering
- Extracted time-based features from the date index:
  - `Year`
  - `Month`
  - `DayOfWeek`

### 3. Model Training
- **Target variable:** `Close` (daily closing price)
- **Train/Test split:** 80% / 20%
- Trained two models:
  - Decision Tree Regressor
  - Random Forest Regressor

### 4. Evaluation
Models were evaluated using:
- **MAE** (Mean Absolute Error)
- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Squared Error)
- **R²** (Coefficient of Determination)

### 5. Visualization
- Plotted actual vs. predicted closing prices using Random Forest

---

## 📊 Model Results

| Metric | Decision Tree | Random Forest |
|--------|--------------|---------------|
| MAE | ~0.347 | ~0.296 |
| MSE | ~0.269 | ~0.264 |
| RMSE | ~0.519 | ~0.514 |
| R² | ~0.9997 | ~0.9997 |

> ✅ **Random Forest outperformed Decision Tree** with lower MAE and RMSE, making it the better model for this task.

---

## 📉 Visualizations

- **Airbnb Stock Closing Price Over Time** — Shows historical price trends
- **Actual vs. Predicted Closing Price** — Demonstrates how well the Random Forest model tracks real stock movements

---

## 🚀 How to Run

1. Clone this repository or download the notebook
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. Open the Jupyter Notebook and run all cells sequentially

---

## 🔍 Key Insights

- The Random Forest model achieves an **R² score of ~0.9997**, indicating near-perfect prediction accuracy on this dataset
- Time-based features (Year, Month, Day of Week) combined with OHLCV data provide strong predictive signals
- The model effectively captures both the overall trend and short-term fluctuations in Airbnb's stock price

---

## ⚠️ Disclaimer

This project is for **educational purposes only**. Stock price predictions made by machine learning models should **not** be used as financial advice or for real trading decisions.

---

## 👤 Author

**Arifudheen**  
GitHub: [github.com/Arif-1411](https://github.com/Arif-1411)
