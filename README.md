# Advanced Time Series Forecasting with LSTM, SARIMA & SHAP

This repository contains my Cultus Skills Center project **“Advanced Time Series Forecasting with Neural Networks and Explainability.”**

The project implements a complete end-to-end time series forecasting pipeline using:
- **LSTM Deep Learning Model**
- **SARIMA Statistical Baseline**
- **SHAP Explainability**
- **Synthetic Multivariate Energy Consumption Dataset**
- **Feature Engineering, Hyperparameter Tuning, and Model Evaluation**

The entire workflow strictly follows the tasks and requirements from the assessment PDF.

---

## 📘 Project Overview

This project forecasts future **energy load** using a **synthetic multivariate time series** with 2000+ hourly observations.

The pipeline includes:

### ✅ Dataset
- Programmatically generated multivariate dataset
- Features:
  - `load` (target)
  - `temperature`
  - `humidity`
  - `hour`
  - `day_of_week`
  - `is_weekend`
- Lags and rolling statistics:
  - `load_lag_1`, `load_lag_2`, `load_lag_3`, `load_lag_24`
  - `load_rollmean_3`, `load_rollmean_24`

### ✅ Models
#### **1. LSTM (Deep Learning Model)**
- Sequence length: 24 hours
- Hyperparameter tuning using **KerasTuner**
- Training with early stopping
- Evaluation on test data

#### **2. SARIMA (Baseline Model)**
- SARIMA(2,1,2) × (1,1,1,24)
- Trained on unscaled target
- Compared against LSTM using same test window

### ✅ Explainability (SHAP)
- SHAP DeepExplainer applied to LSTM model
- Feature importance aggregated over time
- Bar plot & summary plot generated

---

## 📊 Evaluation Metrics
Models are evaluated using:

- **RMSE (Root Mean Squared Error)**
- **MAE (Mean Absolute Error)**
- **MAPE (%)**

The metrics are printed when running the script.

---

## 🗂 Repository Structure

