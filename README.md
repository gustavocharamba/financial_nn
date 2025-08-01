# 🧠 Financial Market Neural Network Trend Forecasting Using Multi-Source Time Series Data

A neural network-based forecasting system for financial markets that integrates multiple sources of historical economic and financial data. This project tackles the challenges of working with time series of varying start dates and frequencies, leveraging an ensemble of neural networks to enhance predictive performance. The system acts as the decision-making engine of an automated trading bot that operates based on consensus signals.

---

## 📄 Abstract

This project introduces a novel approach to financial trend prediction by combining heterogeneous time series — including macroeconomic indicators and technical market metrics — to forecast short-term trends in the **Bitcoin** market.

Two specialized neural networks are used:
- A **Macroeconomic Model**, focused on long-term signals from economic data.
- A **Technical Model**, focused on short-term signals from market indicators.

Initial experiments with a unified model suffered from high loss and instability due to multicollinearity and noisy features. An **ensemble approach** was adopted, where predictions are only acted upon when both models agree (consensus). Backtesting revealed improved accuracy and reduced risk exposure.

---

## ⚙️ Main Features

- **📊 Multi-Frequency Data Integration**  
  Combines macroeconomic indicators (CPI, IPI, Unemployment Rate) with financial metrics (Bitcoin, S&P 500, NASDAQ).

- **📈 Technical Indicators**  
  Includes ADX, SMA, ATR, Bollinger Bands, MACD, RSI, OBV, Parabolic SAR, Stochastic, and VWAP.

- **🕰️ Historical Series Handling**  
  Addresses different data start periods (1960–2010) using alignment techniques such as forward-filling.

- **📉 Trend Prediction**  
  Predicts market direction (**UP/DOWN**) instead of absolute price values.

- **🧩 Ensemble Approach**  
  Separate neural networks for macroeconomic and technical data, with a consensus mechanism for reliable trading signals.

- **🤖 Automated Trading Bot**  
  Executes trades only on model agreement, featuring position sizing and stop-loss rules for risk management.

---

## 📚 Data Sources

- **Economic Indicators**  
  - CPI, Core CPI, Federal Funds Rate  
  - IPI, Payroll, PPI, Unemployment

- **Financial Indicators**  
  - Yield Curves (5Y, 10Y, 30Y)  
  - M1 and M2 Money Supply

- **Market Indexes**  
  - S&P 500, NASDAQ, Bitcoin

---

## 🧪 Methodology

### First Attempt: Unified Model
- Combined all features into one neural network.
- Result: High loss (e.g., 148.59) and unstable accuracy (11.90%–65.08% across folds).
- Problem: Multicollinearity and feature noise.

### Second Attempt: Ensemble Model
- **Economic Neural Network:**  
  Processes ~16 macroeconomic features for long-term trend prediction.

- **Technical Neural Network:**  
  Processes ~22 technical indicators for short-term prediction.

- **Consensus Mechanism:**  
  Trade only when both networks predict the same direction (e.g., both say "UP").

- **Validation Strategy:**  
  Rolling window cross-validation (`window=1260 days`, `step=126 days`) to preserve time order.

#### Neural Network Architecture (per model)
- 3 Dense layers: `512 → 256 → 128 neurons`  
- Activation: ReLU  
- Regularization: Dropout (0.3–0.5)  
- Optimizer: Adam (`lr=0.001`)  
- Loss: Sparse Categorical Cross-Entropy

---

## 📊 Experimental Results

| Model           | Accuracy | Notes                             |
|----------------|----------|-----------------------------------|
| Unified Model  | 11.90%–65.08% | High variance, high loss         |
| Economic NN    | ~60%     | Stable across folds               |
| Technical NN   | ~55%     | Effective in short-term detection |
| **Consensus**  | ~70%     | Improved precision, lower drawdown |

- **Drawdown:**  
  - Ensemble strategy: ~15%  
  - Unified model: ~25%

---

## 📌 Future Work

- Incorporate sentiment analysis from news and social media.
- Explore LSTM or Transformer-based architectures for temporal learning.
- Dynamic weighting of model predictions based on recent performance.

---

## 🧠 Authors & Contributions

- Gustavo Charamba – Data engineering, modeling, bot integration.
