# Stock-Predictor

This project is an end-to-end machine learning and data analysis pipeline for predicting stock prices using historical data from **ADANIPORTS**. It covers exploratory data analysis, technical indicators, ML model training, backtesting strategies for visualization and experimentation.

---

##  Contents

-  Data Preprocessing & Feature Engineering  
-  Machine Learning Models  
-  Technical Indicators (RSI, MACD, SMA, EMA)  
-  Backtesting Strategy  
-  Interactive Streamlit Dashboard  
-  Random Year Prediction  
-  Tools & Libraries  

---

##  Dataset

- CSV File: `ADANIPORTS (1).csv`
- Columns: Date	Symbol	Series	Prev Close	Open	High	Low	Last	Close	VWAP	Volume	Turnover	Trades	Deliverable Volume	%Deliverble
- Source: Historical stock data for **ADANIPORTS**

---

##  Feature Engineering

- **SMA (Simple Moving Average)** – 20-day average to smooth volatility.
- **EMA (Exponential Moving Average)** – 20-day exponentially weighted average.
- **Lag Features** – Previous 1, 2, and 3 day closing prices.
- **Target Variable** – Next day's close price for supervised learning.
- **MACD (Moving Average Convergence Divergence)** – Trend-following indicator.
- **RSI (Relative Strength Index)** – Momentum oscillator.

---

##  Machine Learning Models Used

| Model Type          | Description                              |
|---------------------|------------------------------------------|
|  Linear Regression | Baseline model for trend prediction      |
|  Random Forest     | Tree-based ensemble for classification   |
|  XGBoost           | Gradient-boosted trees for regression    |
|  Signal Strategy   | Rule-based signal generation for trading |

---

##  Backtesting Strategy

A rule-based portfolio strategy was implemented:

- **Buy**: If price expected to go up > 1%
- **Sell**: If price expected to fall > 1%
- Portfolio is updated iteratively, tracking:
  -  Portfolio Value Over Time  
  -  Total Return  
  -  Sharpe Ratio

---

##  Random Year Price Prediction

- Predict closing prices for a **randomly selected year** using Linear Regression.
- Train on all previous years.
- Evaluate prediction accuracy visually.

### Example:

```python
target_year = 2022  # Random year from available data
```

Plots actual vs predicted close price with evaluation table.

---

##  Visualizations

- Correlation Heatmaps
- RSI and MACD Charts
- Portfolio Growth Curve
- Actual vs Predicted Close Price (Line Charts)
- Bar & Pie Charts for distribution and frequency analysis

---

##  Libraries Used

- `pandas`, `numpy` – Data manipulation  
- `matplotlib`, `seaborn` – Visualizations  
- `sklearn` – Machine learning models  
- `xgboost` – Boosted trees  
- `streamlit` – Interactive dashboard  

---

##  How to Run

1. Clone the repo  
2. Install dependencies  
   ```bash
   pip install -r requirements.txt
   ```
3. Launch dashboard  
   ```bash
   streamlit run dashboard.py
   ```
4. Or run scripts manually via Jupyter Notebook / Python scripts

---

##  Highlights

-  **Dynamic stock prediction for a selected year**
-  **ML-powered trading signals**
-  **Interactive UI for stock exploration**
-  **Backtest strategy simulation with portfolio performance**
-  **Modular, extensible, and beginner-friendly codebase**

---

## 📌 To-Do (Future Work)

- Add LSTM and Attention-based models
- Integrate real-time prediction with Yahoo Finance API
- Use Optuna/GridSearchCV for hyperparameter tuning
- Build advanced strategy evaluator
