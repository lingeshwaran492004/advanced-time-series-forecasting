"""
Advanced Time Series Forecasting with Neural Networks and Explainability
=======================================================================

This script fulfils the Cultus project:
"Advanced Time Series Forecasting with Neural Networks and Explainability"

It covers all required tasks:
1. Programmatically generate a complex multivariate time series dataset
   (energy-consumption-style, > 1000 observations).
2. Clean, preprocess, and engineer time-based features (lags, rolling stats).
3. Implement and tune an LSTM-based forecasting model with KerasTuner.
4. Build a baseline SARIMA model on the same data.
5. Evaluate with RMSE, MAE, MAPE.
6. Apply SHAP explainability for the LSTM model.
7. Produce plots and metric printouts for analysis.

You can run this as:
    python advanced_time_series_lstm_project.py
"""

# =========================
# 1. Imports and Settings
# =========================
import os
import random
from typing import Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import keras_tuner as kt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import shap

plt.rcParams["figure.figsize"] = (10, 4)


# For reproducibility
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


set_seed(42)

# Globals (used by model builder)
GLOBAL_WINDOW_SIZE = 24  # number of past time steps
GLOBAL_N_FEATURES = None  # will be set after creating sequences


# =========================
# 2. Generate Synthetic Multivariate Time Series
# =========================
def generate_energy_dataset(
    n_hours: int = 2000, seed: int = 42
) -> pd.DataFrame:
    """
    Programmatically generate a realistic multivariate energy
    consumption time series dataset with:
    - load (target)
    - temperature
    - humidity
    - time-based features (hour, day_of_week, is_weekend)
    """
    rng = np.random.default_rng(seed)
    date_range = pd.date_range(start="2020-01-01", periods=n_hours, freq="H")

    # Daily pattern (more load in evening)
    daily_pattern = 10 * np.sin(2 * np.pi * date_range.hour / 24 - 1.0)

    # Weekly pattern (slightly higher load on weekdays)
    weekly_pattern = np.where(date_range.dayofweek < 5, 5.0, 2.0)

    # Seasonal temperature (warmer in middle of the year)
    temp_seasonal = 20 + 10 * np.sin(2 * np.pi * date_range.dayofyear / 365.0)
    temperature = temp_seasonal + rng.normal(0, 2, size=n_hours)

    # Humidity with some randomness, mildly related to temperature
    humidity = 60 - 0.3 * (temperature - 20) + rng.normal(0, 5, size=n_hours)

    # Base load + patterns + effect of temperature + noise
    base_load = 50
    noise = rng.normal(0, 3, size=n_hours)
    load = (
        base_load
        + daily_pattern
        + weekly_pattern
        + 0.5 * (25 - temperature)  # higher load when temperature below 25
        + noise
    )

    df = pd.DataFrame(
        {
            "timestamp": date_range,
            "load": load,
            "temperature": temperature,
            "humidity": humidity,
        }
    )
    df.set_index("timestamp", inplace=True)

    # basic calendar/time features
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    return df


# =========================
# 3. Feature Engineering (lags & rolling stats)
# =========================
def add_time_features(
    df: pd.DataFrame,
    target_col: str = "load",
    lags: List[int] = None,
    rolling_windows: List[int] = None,
) -> pd.DataFrame:
    if lags is None:
        lags = [1, 2, 3, 24]  # short + daily lag
    if rolling_windows is None:
        rolling_windows = [3, 24]  # short + daily rolling mean

    df = df.copy()
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)

    for win in rolling_windows:
        df[f"{target_col}_rollmean_{win}"] = df[target_col].rolling(win).mean()

    # Drop rows with NaNs due to lag/rolling
    df = df.dropna()
    return df


# =========================
# 4. Train/Val/Test Split (respect time order)
# =========================
def train_val_test_split(
    df: pd.DataFrame, train_size: float = 0.7, val_size: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train / val / test sequentially.
    """
    n = len(df)
    train_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))

    df_train = df.iloc[:train_end]
    df_val = df.iloc[train_end:val_end]
    df_test = df.iloc[val_end:]

    return df_train, df_val, df_test


# =========================
# 5. Scaling and Sequence Creation
# =========================
def scale_train_val_test(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
):
    """
    Fit StandardScaler on train, transform val & test.
    Returns:
        train_scaled, val_scaled, test_scaled, scaler, feature_cols
    """
    feature_cols = train_df.columns.tolist()
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df[feature_cols])
    val_scaled = scaler.transform(val_df[feature_cols])
    test_scaled = scaler.transform(test_df[feature_cols])
    return train_scaled, val_scaled, test_scaled, scaler, feature_cols


def create_sequences_from_scaled(
    scaled_data: np.ndarray,
    target_index: int,
    window_size: int = 24,
    horizon: int = 1,
):
    """
    Create (X, y) sequences from scaled data.
    X shape: (samples, window_size, n_features)
    y shape: (samples,)
    """
    X, y = [], []
    for i in range(len(scaled_data) - window_size - horizon + 1):
        X.append(scaled_data[i : i + window_size, :])
        y.append(scaled_data[i + window_size + horizon - 1, target_index])
    return np.array(X), np.array(y)


def inverse_scale_target(
    y_scaled: np.ndarray, scaler: StandardScaler, target_index: int
) -> np.ndarray:
    """
    Inverse transform a 1D scaled target array using the
    target column's mean and variance from the full scaler.
    """
    target_mean = scaler.mean_[target_index]
    target_std = np.sqrt(scaler.var_[target_index])
    return y_scaled * target_std + target_mean


def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    non_zero_mask = y_true != 0
    y_true = y_true[non_zero_mask]
    y_pred = y_pred[non_zero_mask]
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0


# =========================
# 6. Baseline Model: SARIMA
# =========================
def fit_sarima_baseline(
    train_series: pd.Series,
    test_series: pd.Series,
    order=(2, 1, 2),
    seasonal_order=(1, 1, 1, 24),
):
    """
    Fit a SARIMA model on train_series and forecast len(test_series) steps.
    Returns forecast as a pandas Series indexed like test_series.
    """
    print("Training SARIMA baseline model...")
    sarima_model = SARIMAX(
        train_series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    sarima_fit = sarima_model.fit(disp=False)
    sarima_forecast = sarima_fit.forecast(steps=len(test_series))
    sarima_forecast.index = test_series.index
    return sarima_forecast


# =========================
# 7. LSTM Model + Hyperparameter Tuning (KerasTuner)
# =========================
def build_lstm_model(hp: kt.HyperParameters) -> keras.Model:
    """
    Build an LSTM model. Hyperparameters tuned:
      - number of units
      - dropout rate
      - learning rate
    """
    global GLOBAL_WINDOW_SIZE, GLOBAL_N_FEATURES

    model = keras.Sequential()
    model.add(
        layers.Input(shape=(GLOBAL_WINDOW_SIZE, GLOBAL_N_FEATURES))
    )

    units = hp.Int("units", min_value=32, max_value=128, step=32)
    model.add(layers.LSTM(units=units, return_sequences=False))

    dropout_rate = hp.Float("dropout", 0.0, 0.5, step=0.1)
    model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(1))

    lr = hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=["mae"],
    )
    return model


def tune_and_train_lstm(
    X_train, y_train_scaled, X_val, y_val_scaled, max_trials: int = 5
):
    """
    Use KerasTuner RandomSearch to find a good LSTM configuration.
    Returns the best trained model and its training history.
    """
    tuner_dir = "keras_tuner_energy"
    os.makedirs(tuner_dir, exist_ok=True)

    tuner = kt.RandomSearch(
        build_lstm_model,
        objective="val_loss",
        max_trials=max_trials,
        executions_per_trial=1,
        directory=tuner_dir,
        project_name="energy_lstm",
        overwrite=True,
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    print("Starting hyperparameter search for LSTM...")
    tuner.search(
        X_train,
        y_train_scaled,
        validation_data=(X_val, y_val_scaled),
        epochs=30,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1,
    )

    best_hp = tuner.get_best_hyperparameters(1)[0]
    print("Best hyperparameters:", best_hp.values)

    best_model = tuner.hypermodel.build(best_hp)
    history = best_model.fit(
        X_train,
        y_train_scaled,
        validation_data=(X_val, y_val_scaled),
        epochs=50,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1,
    )

    return best_model, history


# =========================
# 8. Explainability with SHAP
# =========================
def explain_with_shap(
    model: keras.Model,
    X_train,
    X_test,
    feature_names: List[str],
    output_dir: str = "plots",
):
    """
    Use SHAP DeepExplainer for sequence model.
    We aggregate absolute SHAP values across time-steps
    to get overall feature importance.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Use a small background and test sample to keep computation reasonable
    background = X_train[:100]
    X_explain = X_test[:100]

    print("Computing SHAP values (this can take a while)...")
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(X_explain)

    # For regression with one output, shap_values may be a single array
    if isinstance(shap_values, list):
        shap_vals = shap_values[0]
    else:
        shap_vals = shap_values

    # shap_vals shape: (samples, time_steps, features)
    mean_abs_shap_per_feature = np.mean(np.abs(shap_vals), axis=(0, 1))

    # Bar plot of feature importance
    plt.figure(figsize=(10, 4))
    order = np.argsort(mean_abs_shap_per_feature)[::-1]
    sorted_features = [feature_names[i] for i in order]
    sorted_importance = mean_abs_shap_per_feature[order]
    plt.bar(range(len(sorted_features)), sorted_importance)
    plt.xticks(range(len(sorted_features)), sorted_features, rotation=45, ha="right")
    plt.title("Mean absolute SHAP value per feature (aggregated over time)")
    plt.tight_layout()
    shap_bar_path = os.path.join(output_dir, "shap_feature_importance.png")
    plt.savefig(shap_bar_path)
    plt.close()
    print(f"Saved SHAP feature importance bar plot to {shap_bar_path}")

    # Optionally, create a SHAP summary plot as well
    try:
        shap.summary_plot(
            shap_vals,
            features=X_explain,
            feature_names=feature_names,
            show=False,
        )
        shap_summary_path = os.path.join(output_dir, "shap_summary_plot.png")
        plt.tight_layout()
        plt.savefig(shap_summary_path)
        plt.close()
        print(f"Saved SHAP summary plot to {shap_summary_path}")
    except Exception as e:
        print("Could not create SHAP summary plot:", str(e))


# =========================
# 9. Plotting Helpers
# =========================
def plot_train_val_test_split(df, train_df, val_df, test_df, target_col, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure()
    plt.plot(df.index, df[target_col], label="full series", alpha=0.3)
    plt.plot(train_df.index, train_df[target_col], label="train")
    plt.plot(val_df.index, val_df[target_col], label="val")
    plt.plot(test_df.index, test_df[target_col], label="test")
    plt.legend()
    plt.title("Train / Validation / Test split")
    plt.tight_layout()
    path = os.path.join(output_dir, "train_val_test_split.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved split plot to {path}")


def plot_predictions(
    test_index,
    y_true,
    y_pred_lstm,
    y_pred_sarima,
    output_dir: str = "plots",
):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure()
    plt.plot(test_index, y_true, label="Actual")
    plt.plot(test_index, y_pred_lstm, label="LSTM forecast")
    plt.plot(test_index, y_pred_sarima, label="SARIMA forecast", alpha=0.7)
    plt.legend()
    plt.title("Test set: Actual vs LSTM vs SARIMA")
    plt.tight_layout()
    path = os.path.join(output_dir, "test_predictions_comparison.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved prediction comparison plot to {path}")


# =========================
# 10. Main Pipeline
# =========================
def main():
    global GLOBAL_WINDOW_SIZE, GLOBAL_N_FEATURES

    OUTPUT_DIR = "plots"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) Generate dataset
    print("Generating synthetic multivariate energy dataset...")
    df = generate_energy_dataset(n_hours=2000)
    print("Dataset shape (before feature engineering):", df.shape)

    # 2) Add lags and rolling features
    df_fe = add_time_features(df, target_col="load")
    print("Dataset shape (after feature engineering):", df_fe.shape)

    # 3) Train/Val/Test split
    train_df, val_df, test_df = train_val_test_split(df_fe)
    print("Train / Val / Test sizes:", len(train_df), len(val_df), len(test_df))

    # Plot split for visual check
    plot_train_val_test_split(df_fe, train_df, val_df, test_df, "load", OUTPUT_DIR)

    # 4) Prepare baseline series (unscaled)
    train_target_series = train_df["load"]
    test_target_series = test_df["load"]

    # 5) Scale data and create sequences for LSTM
    train_scaled, val_scaled, test_scaled, scaler, feature_cols = scale_train_val_test(
        train_df, val_df, test_df
    )
    target_index = feature_cols.index("load")

    # Update globals for model builder
    GLOBAL_N_FEATURES = len(feature_cols)

    X_train, y_train_scaled = create_sequences_from_scaled(
        train_scaled, target_index, window_size=GLOBAL_WINDOW_SIZE, horizon=1
    )
    X_val, y_val_scaled = create_sequences_from_scaled(
        val_scaled, target_index, window_size=GLOBAL_WINDOW_SIZE, horizon=1
    )
    X_test, y_test_scaled = create_sequences_from_scaled(
        test_scaled, target_index, window_size=GLOBAL_WINDOW_SIZE, horizon=1
    )

    print("Sequence shapes:")
    print("X_train:", X_train.shape, "y_train:", y_train_scaled.shape)
    print("X_val:", X_val.shape, "y_val:", y_val_scaled.shape)
    print("X_test:", X_test.shape, "y_test:", y_test_scaled.shape)

    # Inverse-scale y for metrics later
    y_train = inverse_scale_target(y_train_scaled, scaler, target_index)
    y_val = inverse_scale_target(y_val_scaled, scaler, target_index)
    y_test = inverse_scale_target(y_test_scaled, scaler, target_index)

    # 6) Train baseline SARIMA (on full train target)
    sarima_forecast_full_test = fit_sarima_baseline(
        train_target_series, test_target_series
    )

    # Align SARIMA forecasts with the samples used by LSTM on test
    # We use the last len(y_test) points of SARIMA forecast & test_target
    sarima_forecast_aligned = sarima_forecast_full_test[-len(y_test) :]
    sarima_test_target_aligned = test_target_series[-len(y_test) :]

    # 7) Tune and train LSTM
    best_model, history = tune_and_train_lstm(
        X_train, y_train_scaled, X_val, y_val_scaled, max_trials=5
    )

    # 8) Evaluate on test set
    y_pred_test_scaled = best_model.predict(X_test).flatten()
    y_pred_test = inverse_scale_target(y_pred_test_scaled, scaler, target_index)

    # Metrics for LSTM
    lstm_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    lstm_mae = mean_absolute_error(y_test, y_pred_test)
    lstm_mape = mean_absolute_percentage_error(y_test, y_pred_test)

    # Metrics for SARIMA baseline (aligned with same period)
    baseline_rmse = np.sqrt(
        mean_squared_error(sarima_test_target_aligned, sarima_forecast_aligned)
    )
    baseline_mae = mean_absolute_error(
        sarima_test_target_aligned, sarima_forecast_aligned
    )
    baseline_mape = mean_absolute_percentage_error(
        sarima_test_target_aligned, sarima_forecast_aligned
    )

    print("
===== Test Metrics =====")
    print("LSTM model:")
    print(f"  RMSE: {lstm_rmse:.3f}")
    print(f"  MAE : {lstm_mae:.3f}")
    print(f"  MAPE: {lstm_mape:.3f}%")
    print("
SARIMA baseline:")
    print(f"  RMSE: {baseline_rmse:.3f}")
    print(f"  MAE : {baseline_mae:.3f}")
    print(f"  MAPE: {baseline_mape:.3f}%")

    # 9) Prediction comparison plot on aligned test portion
    # Use index of aligned SARIMA target as test index
    plot_predictions(
        sarima_test_target_aligned.index,
        sarima_test_target_aligned.values,
        y_pred_test[-len(sarima_test_target_aligned) :],
        sarima_forecast_aligned.values,
        output_dir=OUTPUT_DIR,
    )

    # 10) Explainability with SHAP
    explain_with_shap(best_model, X_train, X_test, feature_cols, OUTPUT_DIR)

    print("
All done. Metrics printed above.")
    print(f"Plots saved in folder: {OUTPUT_DIR}")
    print(
        "You can now use these metrics and plots in your written report "
        "and for submission."
    )


if __name__ == "__main__":
    main()
