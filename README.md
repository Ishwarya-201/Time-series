# Time-series
# Advanced Time Series Forecasting with Attention-LSTM
This project implements **advanced time series forecasting** using **deep learning with an Attention mechanism** on top of an LSTM backbone. It is designed to fully meet the given assignment requirements:
1. Programmatically generate a **synthetic, multivariate, non-stationary time series** (≥ 1000 observations, 3 features) with clear trend, seasonality, and noise.
2. Implement a **custom Deep Learning model** (Attention-based LSTM) in **PyTorch**, with production-style, well-documented code.
3. Train and compare the Attention model with **two baselines**:
   - Vanilla LSTM (no attention)
   - ARIMA (classical statistical model, with order selection)
4. Perform forecasting **evaluation and comparison** using:
   - RMSE, MAE, MAPE
   - **Diebold–Mariano test** for statistical significance of performance differences.
5. Analyze and **visualize learned attention weights** to interpret how the model uses historical time steps.

## 1. Project Overview
Modern time series often show complex patterns: trend, multiple seasonalities, and changing noise (non-stationarity). Traditional models such as ARIMA can struggle with this complexity.
This project:
- **Simulates** a realistic time series with:
  - Quadratic trend  
  - Daily and weekly seasonality  
  - Time-varying volatility (noise level changes over time)
- Uses an **LSTM with self-attention** to learn which past time steps are most relevant for forecasting.
- Benchmarks against:
  - A **Vanilla LSTM** (same backbone, no attention)
  - **ARIMA** (with automatic (p, d, q) order selection using AIC)
- Applies the **Diebold–Mariano test** to check if performance differences are statistically significant.
  
## 2. Features
-  Synthetic **3-feature multivariate** time series (3000 points)
-  **Non-stationary**: trend + changing noise + multi-seasonality
-  **Attention-LSTM** model (PyTorch), with multi-head self-attention
-  Baseline models:
  - Vanilla LSTM
  - ARIMA with order tuning
-  Hyperparameter search for LSTM models
-  Evaluation metrics:
  - MAE, RMSE, MAPE
-  **Diebold–Mariano test** to compare model forecasts
-  Attention heatmaps to show learned temporal focus

## 3. File Structure
If you keep everything in a single script, your repository might look like this:
├── advanced_time_series_attention.py   
├── README.md                           
└── requirements.txt                   
