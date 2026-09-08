# -*- coding: utf-8 -*-
"""
Bitcoin Forecasting V6.2 — Publication-Grade
======================

FULL-DATA, LEAKAGE-AWARE, DIRECT MULTI-HORIZON WALK-FORWARD STUDY

Purpose
-------
This is the publication-oriented V6 implementation of the Bitcoin forecasting
study developed through V1-V5. It replaces the 300-origin pilot with ALL
valid forecast origins in the frozen modeling dataset.

Primary research question
-------------------------
Does model superiority depend on forecast horizon and market regime?

Horizons
--------
15D, 1M, 2M, 3M, 6M, 12M, 24M, 36M

Target
------
Direct cumulative log return:
    R(t,h) = 100 * log(P[t+h] / P[t])

No recursive forecasts are used.

Models
------
1. Zero Return benchmark
2. Historical Mean
3. Random Walk with Drift (same return forecast as historical mean)
4. Historical Median
5. Majority Direction
6. Persistence Direction
7. ARIMA
8. OLS
9. GLS
10. LSTM
11. VS-LSTM (volatility-scaled/conditioned LSTM; deliberately not called
    standard GARCH-LSTM)
12. GLS-LSTM residual hybrid
13. Adaptive Ensemble

Important methodological safeguards
-----------------------------------
- Expanding-window walk-forward evaluation.
- First valid origin = INITIAL_TRAIN + horizon.
- At origin t, training targets use only observations whose t+h target
  is already observable: target rows <= t-1.
- Features are restricted to information available at the forecast origin.
- Direct multi-horizon targets; no recursive accumulation of forecasts.
- Retraining every RETRAIN_EVERY origins.
- Causal adaptive-ensemble weights based only on earlier OOS errors.
- No MAPE on returns.
- Price metrics are reconstructed only after return prediction.
- Overlap-aware HAC DM test.
- HLN-style small-sample correction is reported as a sensitivity, not as a
  replacement for HAC.
- MCS at alpha = 0.05, 0.10, 0.15.
- Moving-block bootstrap diagnostics for long-horizon / non-overlap issues.
- Positive-return base-rate analysis is explicitly separated from
  conditional predictability.
- Wilson confidence intervals + exact binomial test for positive-rate.
- Calendar-period positive-rate diagnostics.
- Skill relative to Historical Mean / Drift and Zero Return.
- Computational audit and reproducibility metadata.
- Checkpointing after every horizon/model batch so a long run can resume.
- Primary full run uses one prespecified seed. A separate optional
  multi-seed sensitivity module is included and does NOT multiply the
  computational cost of the main full run.

Requirements
------------
Python 3.10+
pandas, numpy, scipy, statsmodels, scikit-learn, matplotlib
tensorflow (for LSTM models)

The script is designed to run on CPU as well as GPU. Native Windows
TensorFlow GPU availability depends on the installed TensorFlow stack.

Input
-----
Default:
E:\\Daneshgahi\\Pazhoheshi\\New Study\\sohrabati\\renew paper\\modeling_data\\bitcoin_modeling_dataset.csv

Outputs
-------
forecasting_results_V6_2/
forecasting_figures_V6_2/
forecasting_logs_V6_2/

The script writes:
- full forecast-level results
- horizon/model metrics
- MCS sensitivity
- pairwise HAC-DM
- sign-distribution diagnostics
- calendar-period positive-rate diagnostics
- skill scores
- robustness diagnostics
- computational audit
- reproducibility metadata
- summary
- checkpoint files

Author note
-----------
This code intentionally treats 100% positive long-horizon realized returns
as a property of the evaluation sample that requires diagnosis. It does not
interpret 100% positive realized returns as evidence of 100% forecastability.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import scipy
from scipy import stats
from statsmodels.stats.diagnostic import breaks_cusumolsresid

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

warnings.filterwarnings("ignore")

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_DIR / "data" / "processed" / "bitcoin_modeling_dataset.csv"

RESULT_DIR = PROJECT_DIR / "forecasting_results_V6_2"
FIGURE_DIR = PROJECT_DIR / "forecasting_figures_V6_2"
LOG_DIR = PROJECT_DIR / "forecasting_logs_V6_2"
CHECKPOINT_DIR = RESULT_DIR / "checkpoints"

for d in [RESULT_DIR, FIGURE_DIR, LOG_DIR, CHECKPOINT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

HORIZONS = {
    "15D": 15,
    "1M": 30,
    "2M": 60,
    "3M": 90,
    "6M": 180,
    "12M": 365,
    "24M": 730,
    "36M": 1095,
}

INITIAL_TRAIN = 1000
RETRAIN_EVERY = 25

# FULL RUN: use every valid origin. Set to an integer only for a local pilot.
MAX_FORECASTS = None

# Main reproducibility seed. Do not change merely to use the conventional 42.
RANDOM_SEED = 20260819

# MCS sensitivity.
MCS_ALPHAS = [0.05, 0.10, 0.15]
MCS_BOOTSTRAPS = 1000
MCS_BLOCK_LENGTH = 10

# Long-horizon robustness diagnostics.
ROBUSTNESS_BOOTSTRAPS = 10000
# V6.2 stability / checkpoint controls.
# Partial horizon results are saved during the run and can be resumed.
CHECKPOINT_EVERY = 100
SAVE_AFTER_EVERY_RETRAIN = True

# V6.2 uses a single primary full-data run by default. The bounded
# multi-seed sensitivity can be enabled explicitly after the primary run.

BOOTSTRAP_BLOCK_LENGTHS = [5, 10, 20]

# Optional multi-seed sensitivity for stochastic neural models.
# This is NOT multiplied into the primary full run.
RUN_MULTI_SEED_SENSITIVITY = False
RUN_RETRAIN_SENSITIVITY = False
RETRAIN_SENSITIVITY_INTERVALS = [10, 25, 50]
RETRAIN_SENSITIVITY_HORIZONS = ["15D", "3M", "12M"]
RETRAIN_SENSITIVITY_MAX_ORIGINS = 100
MULTI_SEEDS = [20260819, 20260820, 20260821, 20260822, 20260823]
MULTI_SEED_HORIZONS = ["15D", "3M", "12M"]
MULTI_SEED_MAX_ORIGINS = 100

# LSTM.
LSTM_LOOKBACK = 30
LSTM_EPOCHS = 40
LSTM_BATCH_SIZE = 32
LSTM_PATIENCE = 6
LSTM_VERBOSE = 0

# ARIMA.
ARIMA_ORDER = (1, 0, 1)

# GLSAR.
GLSAR_RHO = 1
GLSAR_MAXITER = 8

# Ensemble.
MIN_ENSEMBLE_HISTORY = 20
ENSEMBLE_EPS = 1e-8

# Regime definition.
ROLLING_VOL_WINDOW = 30

# Metrics.
EPS = 1e-12

# Feature columns that existed in the frozen modeling dataset.
BASE_FEATURES = [
    "active_addresses",
    "transactions",
    "difficulty",
    "hashrate",
    "block_size",
    "transaction_value",
    "avg_transaction_value",
    "price",
    "return_log_pct",
    "return_simple_pct",
    "lag1_price",
    "lag1_return",
    "rolling_vol_7d",
    "rolling_vol_30d",
    "day_of_week",
    "month",
    "year",
]

# Numeric features used by regression/LSTM.
MODEL_FEATURES = [
    "active_addresses",
    "transactions",
    "difficulty",
    "hashrate",
    "block_size",
    "transaction_value",
    "avg_transaction_value",
    "price",
    "return_log_pct",
    "lag1_return",
    "rolling_vol_7d",
    "rolling_vol_30d",
    "day_of_week",
    "month",
    "year",
]

# Volatility-conditioned LSTM features.
# The model retains the same information set as LSTM and explicitly
# exposes the contemporaneous 7-day and 30-day rolling volatility inputs.
VS_LSTM_FEATURES = [
    "active_addresses",
    "difficulty",
    "hashrate",
    "price",
    "return_log_pct",
    "lag1_return",
    "rolling_vol_7d",
    "rolling_vol_30d",
]


# =============================================================================
# 2. REPRODUCIBILITY
# =============================================================================

def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if TF_AVAILABLE:
        try:
            tf.keras.utils.set_random_seed(seed)
            try:
                tf.config.experimental.enable_op_determinism()
            except Exception:
                pass
        except Exception:
            pass


set_global_seed(RANDOM_SEED)

# =============================================================================
# 3. LOGGING
# =============================================================================

LOG_FILE = LOG_DIR / "V6_2_full_run.log"

def log(msg: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


# =============================================================================
# 4. DATA VALIDATION
# =============================================================================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_dataset() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    required = {"date", "price"} | set(MODEL_FEATURES)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)

    if df["date"].isna().any():
        raise ValueError("Invalid date values found.")

    if not (df["price"] > 0).all():
        raise ValueError("Non-positive prices found.")

    for c in MODEL_FEATURES:
        if not np.isfinite(df[c].to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite values found in {c}")

    log(
        f"Loaded dataset: {len(df)} rows, "
        f"{df['date'].min().date()} to {df['date'].max().date()}"
    )
    return df


# =============================================================================
# 5. TARGET CONSTRUCTION
# =============================================================================

def direct_target(df: pd.DataFrame, h: int) -> np.ndarray:
    p = df["price"].to_numpy(dtype=float)
    y = np.full(len(df), np.nan, dtype=float)
    y[:-h] = 100.0 * np.log(p[h:] / p[:-h])
    return y


# =============================================================================
# 6. BENCHMARKS
# =============================================================================

def historical_mean_forecast(y_train: np.ndarray) -> float:
    return float(np.mean(y_train))


def historical_median_forecast(y_train: np.ndarray) -> float:
    return float(np.median(y_train))


def majority_direction_forecast(y_train: np.ndarray) -> float:
    pos = np.sum(y_train > 0)
    neg = np.sum(y_train < 0)
    if pos > neg:
        return abs(float(np.mean(np.abs(y_train))))
    if neg > pos:
        return -abs(float(np.mean(np.abs(y_train))))
    return 0.0


def persistence_direction_forecast(last_return: float, y_train: np.ndarray) -> float:
    mag = abs(float(np.mean(y_train))) if len(y_train) else 0.0
    if last_return > 0:
        return mag
    if last_return < 0:
        return -mag
    return 0.0


# =============================================================================
# 7. STANDARDIZATION / REGRESSION
# =============================================================================

def prepare_X(
    df: pd.DataFrame,
    end_row_exclusive: int
) -> Tuple[np.ndarray, StandardScaler]:
    X = df.loc[:end_row_exclusive - 1, MODEL_FEATURES].copy()

    # Log-transform strongly skewed positive on-chain variables.
    for c in [
        "active_addresses",
        "transactions",
        "difficulty",
        "hashrate",
        "block_size",
        "transaction_value",
        "avg_transaction_value",
        "price",
    ]:
        if c in X.columns:
            X[c] = np.log1p(np.maximum(X[c].astype(float), 0.0))

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return Xs, scaler


def fit_ols_gls(
    df: pd.DataFrame,
    y: np.ndarray,
    train_target_count: int,
    model: str
):
    X_df = df.iloc[:train_target_count][MODEL_FEATURES].copy()
    y_train = y[:train_target_count].astype(float)

    for c in [
        "active_addresses",
        "transactions",
        "difficulty",
        "hashrate",
        "block_size",
        "transaction_value",
        "avg_transaction_value",
        "price",
    ]:
        X_df[c] = np.log1p(np.maximum(X_df[c].astype(float), 0.0))

    scaler = StandardScaler()
    X = scaler.fit_transform(X_df)
    X = sm.add_constant(X, has_constant="add")

    if model == "OLS":
        result = sm.OLS(y_train, X).fit()
    elif model == "GLS":
        result = sm.GLSAR(
            y_train, X, rho=GLSAR_RHO
        ).iterative_fit(maxiter=GLSAR_MAXITER)
    else:
        raise ValueError(model)

    return result, scaler


def predict_regression(
    model,
    scaler,
    df: pd.DataFrame,
    row: int
) -> float:
    xdf = df.iloc[[row]][MODEL_FEATURES].copy()

    for c in [
        "active_addresses",
        "transactions",
        "difficulty",
        "hashrate",
        "block_size",
        "transaction_value",
        "avg_transaction_value",
        "price",
    ]:
        xdf[c] = np.log1p(np.maximum(xdf[c].astype(float), 0.0))

    x = scaler.transform(xdf)
    x = sm.add_constant(x, has_constant="add")
    return float(model.predict(x)[0])


# =============================================================================
# 8. ARIMA
# =============================================================================

def fit_arima(y_train: np.ndarray):
    return ARIMA(y_train, order=ARIMA_ORDER, trend="c").fit()


def predict_arima(model, h: int) -> float:
    """Forecast y[t] from the last observable direct-target y[t-h]."""
    return float(model.forecast(steps=h)[-1])


# =============================================================================
# 9. LSTM UTILITIES
# =============================================================================

def build_sequences(
    X: np.ndarray,
    y: np.ndarray,
    lookback: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Create causal sequences [i-lookback+1, ..., i] -> y[i]."""
    Xs, ys = [], []
    for i in range(lookback - 1, len(X)):
        if np.isfinite(y[i]):
            Xs.append(X[i - lookback + 1:i + 1])
            ys.append(y[i])
    if not Xs:
        return np.empty((0, lookback, X.shape[1])), np.empty((0,))
    return np.asarray(Xs), np.asarray(ys)


def build_lstm(input_shape: Tuple[int, int]) -> "keras.Model":
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.LSTM(32, return_sequences=False),
        keras.layers.Dropout(0.15),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse"
    )
    return model


def _prepare_lstm_features(
    df: pd.DataFrame,
    start: int,
    end: int,
    feature_columns: List[str]
) -> pd.DataFrame:
    """Prepare exactly the feature set used by a fitted LSTM."""
    Xdf = df.iloc[start:end][feature_columns].copy()

    for c in [
        "active_addresses",
        "transactions",
        "difficulty",
        "hashrate",
        "block_size",
        "transaction_value",
        "avg_transaction_value",
        "price",
    ]:
        if c in Xdf.columns:
            Xdf[c] = np.log1p(np.maximum(Xdf[c].astype(float), 0.0))

    return Xdf


def fit_lstm(
    df: pd.DataFrame,
    y: np.ndarray,
    train_target_count: int,
    seed: int,
    target_transform: str = "return",
    feature_columns: Optional[List[str]] = None
):
    if not TF_AVAILABLE:
        return None

    set_global_seed(seed)

    if feature_columns is None:
        feature_columns = MODEL_FEATURES

    Xdf = _prepare_lstm_features(
        df, 0, train_target_count, feature_columns
    )

    scaler = StandardScaler()
    X = scaler.fit_transform(Xdf)

    yy = y[:train_target_count].astype(float)

    Xseq, yseq = build_sequences(X, yy, LSTM_LOOKBACK)
    if len(yseq) < 100:
        return None

    model = build_lstm((Xseq.shape[1], Xseq.shape[2]))

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=LSTM_PATIENCE,
            restore_best_weights=True,
            min_delta=1e-5,
        )
    ]

    history = model.fit(
        Xseq,
        yseq,
        validation_split=0.10,
        epochs=LSTM_EPOCHS,
        batch_size=LSTM_BATCH_SIZE,
        shuffle=False,
        callbacks=callbacks,
        verbose=LSTM_VERBOSE,
    )

    return {
        "model": model,
        "scaler": scaler,
        "history": history.history,
        "seed": seed,
        "epochs_run": len(history.history.get("loss", [])),
        "feature_columns": list(feature_columns),
        "trainable_params": int(model.count_params()),
    }


def predict_lstm(
    fitted,
    df: pd.DataFrame,
    row: int
) -> float:
    if fitted is None:
        return 0.0

    scaler = fitted["scaler"]
    model = fitted["model"]
    feature_columns = fitted.get("feature_columns", MODEL_FEATURES)

    start = max(0, row - LSTM_LOOKBACK + 1)
    Xdf = _prepare_lstm_features(
        df, start, row + 1, feature_columns
    )

    X = scaler.transform(Xdf)

    if len(X) < LSTM_LOOKBACK:
        pad = np.repeat(X[[0]], LSTM_LOOKBACK - len(X), axis=0)
        X = np.vstack([pad, X])

    X = X[-LSTM_LOOKBACK:]

    # V6.2: call the Keras model directly instead of model.predict().
    # This avoids rebuilding tf.function wrappers for thousands of
    # single-sample calls and substantially reduces retracing overhead.
    try:
        pred = model(
            X[np.newaxis, :, :].astype(np.float32),
            training=False
        ).numpy()
        return float(pred[0, 0])
    except Exception:
        return 0.0


# =============================================================================
# 10. VS-LSTM
# =============================================================================

def fit_vs_lstm(
    df: pd.DataFrame,
    y: np.ndarray,
    train_target_count: int,
    seed: int
):
    """
    Volatility-conditioned LSTM.

    This is intentionally named VS-LSTM rather than standard GARCH-LSTM.
    Rolling 7-day and 30-day volatility are explicitly included as
    contemporaneous explanatory variables. No future volatility is used.
    """
    return fit_lstm(
        df,
        y,
        train_target_count,
        seed,
        "return",
        feature_columns=VS_LSTM_FEATURES
    )


# =============================================================================
# 11. GLS-LSTM RESIDUAL HYBRID
# =============================================================================

def fit_gls_lstm_hybrid(
    df: pd.DataFrame,
    y: np.ndarray,
    train_target_count: int,
    seed: int
):
    gls, scaler_gls = fit_ols_gls(df, y, train_target_count, "GLS")

    residual = np.full(train_target_count, np.nan)
    Xdf = df.iloc[:train_target_count][MODEL_FEATURES].copy()
    for c in [
        "active_addresses",
        "transactions",
        "difficulty",
        "hashrate",
        "block_size",
        "transaction_value",
        "avg_transaction_value",
        "price",
    ]:
        Xdf[c] = np.log1p(np.maximum(Xdf[c].astype(float), 0.0))

    X = scaler_gls.transform(Xdf)
    X = sm.add_constant(X, has_constant="add")
    residual = y[:train_target_count] - gls.predict(X)

    # Fit LSTM to the historical GLS residual.
    lstm = fit_lstm(
        df,
        residual,
        train_target_count,
        seed,
        "return"
    )

    return {
        "gls": gls,
        "gls_scaler": scaler_gls,
        "residual_lstm": lstm,
    }


def predict_gls_lstm_hybrid(
    fitted,
    df: pd.DataFrame,
    row: int
) -> float:
    gls_pred = predict_regression(
        fitted["gls"],
        fitted["gls_scaler"],
        df,
        row
    )
    resid_pred = predict_lstm(
        fitted["residual_lstm"],
        df,
        row
    )
    return float(gls_pred + resid_pred)


# =============================================================================
# 12. FIT ALL MODELS AT ONE RETRAINING ORIGIN
# =============================================================================

MODEL_NAMES = [
    "Zero Return",
    "Historical Mean",
    "Random Walk with Drift",
    "Historical Median",
    "Majority Direction",
    "Persistence Direction",
    "ARIMA",
    "OLS",
    "GLS",
    "LSTM",
    "VS-LSTM",
    "GLS-LSTM",
    "Adaptive Ensemble",
]

FITTABLE_MODELS = [
    "ARIMA",
    "OLS",
    "GLS",
    "LSTM",
    "VS-LSTM",
    "GLS-LSTM",
]


def fit_models(
    df: pd.DataFrame,
    y: np.ndarray,
    origin: int,
    h: int
) -> Tuple[Dict, Dict]:
    """
    At origin t, y[t-h] is observable because it only requires prices through t.

    The Python training end is therefore origin-h+1 (exclusive end index),
    so the latest observable direct-target label y[t-h] is included.

    This is the critical direct multi-horizon leakage control.
    """
    train_target_end = origin - h + 1

    y_train = y[:train_target_end]
    y_train = y_train[np.isfinite(y_train)]
    if len(y_train) < INITIAL_TRAIN:
        raise ValueError(
            f"Insufficient leakage-free training data at origin={origin}, "
            f"h={h}, valid_training_targets={len(y_train)}"
        )

    # V6.2 memory safeguard:
    # the previous retraining cycle is no longer needed. Clear its Keras
    # graph before constructing the next set of deep models. This is done
    # ONCE per retraining cycle, not after each model, so the newly fitted
    # models can coexist for OOS prediction.
    if TF_AVAILABLE:
        try:
            tf.keras.backend.clear_session()
        except Exception:
            pass
        gc.collect()

    fitted = {}
    audit = {}

    # Classical models.
    t0 = time.perf_counter()
    try:
        fitted["ARIMA"] = fit_arima(y_train)
    except Exception as exc:
        fitted["ARIMA"] = None
        log(f"ARIMA fit failed at origin {origin}, h={h}: {exc}")
    audit["ARIMA_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    try:
        fitted["OLS"], fitted["OLS_scaler"] = fit_ols_gls(
            df, y, train_target_end, "OLS"
        )
    except Exception as exc:
        fitted["OLS"] = None
        fitted["OLS_scaler"] = None
        log(f"OLS fit failed at origin {origin}, h={h}: {exc}")
    audit["OLS_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    try:
        fitted["GLS"], fitted["GLS_scaler"] = fit_ols_gls(
            df, y, train_target_end, "GLS"
        )
    except Exception as exc:
        fitted["GLS"] = None
        fitted["GLS_scaler"] = None
        log(f"GLS fit failed at origin {origin}, h={h}: {exc}")
    audit["GLS_sec"] = time.perf_counter() - t0

    # LSTM.
    if TF_AVAILABLE:
        t0 = time.perf_counter()
        try:
            fitted["LSTM"] = fit_lstm(
                df, y, train_target_end, RANDOM_SEED
            )
        except Exception as exc:
            fitted["LSTM"] = None
            log(f"LSTM fit failed at origin {origin}, h={h}: {exc}")
        audit["LSTM_sec"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        try:
            fitted["VS-LSTM"] = fit_vs_lstm(
                df, y, train_target_end, RANDOM_SEED + 100
            )
        except Exception as exc:
            fitted["VS-LSTM"] = None
            log(f"VS-LSTM fit failed at origin {origin}, h={h}: {exc}")
        audit["VS-LSTM_sec"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        try:
            fitted["GLS-LSTM"] = fit_gls_lstm_hybrid(
                df, y, train_target_end, RANDOM_SEED + 200
            )
        except Exception as exc:
            fitted["GLS-LSTM"] = None
            log(f"GLS-LSTM fit failed at origin {origin}, h={h}: {exc}")
        audit["GLS-LSTM_sec"] = time.perf_counter() - t0
    else:
        fitted["LSTM"] = None
        fitted["VS-LSTM"] = None
        fitted["GLS-LSTM"] = None
        audit["LSTM_sec"] = np.nan
        audit["VS-LSTM_sec"] = np.nan
        audit["GLS-LSTM_sec"] = np.nan

    return fitted, audit


# =============================================================================
# 13. PREDICTION FROM FITTED MODELS
# =============================================================================

def predict_fitted_models(
    fitted: Dict,
    df: pd.DataFrame,
    y: np.ndarray,
    origin: int,
    h: int,
    history_errors: Dict[str, List[float]]
) -> Dict[str, float]:

    train_target_end = origin - h + 1
    y_train = y[:train_target_end]
    y_train = y_train[np.isfinite(y_train)]

    pred = {}

    pred["Zero Return"] = 0.0
    hm = historical_mean_forecast(y_train)
    pred["Historical Mean"] = hm
    pred["Random Walk with Drift"] = hm
    pred["Historical Median"] = historical_median_forecast(y_train)
    pred["Majority Direction"] = majority_direction_forecast(y_train)

    last_return = float(df.iloc[origin]["return_log_pct"])
    pred["Persistence Direction"] = persistence_direction_forecast(
        last_return, y_train
    )

    if fitted.get("ARIMA") is not None:
        try:
            pred["ARIMA"] = predict_arima(fitted["ARIMA"], h)
        except Exception:
            pred["ARIMA"] = 0.0
    else:
        pred["ARIMA"] = 0.0

    if fitted.get("OLS") is not None:
        pred["OLS"] = predict_regression(
            fitted["OLS"], fitted["OLS_scaler"], df, origin
        )
    else:
        pred["OLS"] = 0.0

    if fitted.get("GLS") is not None:
        pred["GLS"] = predict_regression(
            fitted["GLS"], fitted["GLS_scaler"], df, origin
        )
    else:
        pred["GLS"] = 0.0

    pred["LSTM"] = predict_lstm(fitted.get("LSTM"), df, origin)
    pred["VS-LSTM"] = predict_lstm(fitted.get("VS-LSTM"), df, origin)
    pred["GLS-LSTM"] = predict_gls_lstm_hybrid(
        fitted["GLS-LSTM"], df, origin
    ) if fitted.get("GLS-LSTM") is not None else 0.0

    # Causal adaptive ensemble.
    # Components are magnitude forecasts. Historical Mean and Random Walk with
    # Drift are identical under the present direct-return formulation, so RWD
    # is excluded to avoid double-counting. Direction-only benchmarks are
    # evaluated separately and are not combined with magnitude forecasts.
    eligible = [
        "Zero Return",
        "Historical Mean",
        "Historical Median",
        "ARIMA",
        "OLS",
        "GLS",
        "LSTM",
        "VS-LSTM",
        "GLS-LSTM",
    ]

    fitted_available = {
        "ARIMA": fitted.get("ARIMA") is not None,
        "OLS": fitted.get("OLS") is not None,
        "GLS": fitted.get("GLS") is not None,
        "LSTM": fitted.get("LSTM") is not None,
        "VS-LSTM": fitted.get("VS-LSTM") is not None,
        "GLS-LSTM": fitted.get("GLS-LSTM") is not None,
    }

    available = []
    for m in eligible:
        if m in fitted_available and not fitted_available[m]:
            continue
        errs = history_errors.get(m, [])
        if len(errs) >= MIN_ENSEMBLE_HISTORY:
            recent = np.asarray(errs[-MIN_ENSEMBLE_HISTORY:], dtype=float)
            recent = recent[np.isfinite(recent)]
            if len(recent) >= MIN_ENSEMBLE_HISTORY:
                mae = float(np.mean(np.abs(recent)))
                weight = 1.0 / max(mae, ENSEMBLE_EPS)
                available.append((m, weight))

    if not available:
        base = []
        for m in eligible:
            if m in {"Zero Return", "Historical Mean", "Historical Median"} or fitted_available.get(m, False):
                base.append(m)
        vals = [pred[m] for m in base]
        pred["Adaptive Ensemble"] = float(np.mean(vals)) if vals else 0.0
        ensemble_info = {m: 1.0 / len(base) for m in base} if base else {}
    else:
        weights = np.asarray([x[1] for x in available], dtype=float)
        weights /= weights.sum()
        ensemble_info = {m: float(w) for w, (m, _) in zip(weights, available)}
        pred["Adaptive Ensemble"] = float(
            sum(w * pred[m] for w, (m, _) in zip(weights, available))
        )

    pred["__ensemble_info__"] = json.dumps(ensemble_info, sort_keys=True)

    return pred


# =============================================================================
# 14. FORECAST GENERATION
# =============================================================================

def valid_forecast_origins(
    n: int,
    h: int,
    max_forecasts: Optional[int] = None
) -> np.ndarray:
    # At origin t, y[t-h] = 100*log(P[t]/P[t-h]) is fully observable.
    # Hence the first origin with exactly INITIAL_TRAIN usable target labels is
    # INITIAL_TRAIN + h - 1.
    first = INITIAL_TRAIN + h - 1
    last = n - h - 1

    if last < first:
        return np.array([], dtype=int)

    origins = np.arange(first, last + 1)

    if max_forecasts is not None and len(origins) > max_forecasts:
        origins = origins[:max_forecasts]

    return origins


def checkpoint_config(hname: str, h: int, dataset_sha256: str) -> Dict:
    """
    Configuration fingerprint for safe checkpoint/resume.

    Any change that can alter target alignment, model specification, ensemble
    composition, or forecast-origin selection invalidates an old checkpoint.
    """
    return {
        "version": "V6.2.1",
        "horizon": hname,
        "horizon_days": h,
        "dataset_sha256": dataset_sha256,
        "initial_train": INITIAL_TRAIN,
        "retrain_every": RETRAIN_EVERY,
        "max_forecasts": MAX_FORECASTS,
        "random_seed": RANDOM_SEED,
        "lstm_lookback": LSTM_LOOKBACK,
        "lstm_epochs": LSTM_EPOCHS,
        "lstm_batch_size": LSTM_BATCH_SIZE,
        "lstm_patience": LSTM_PATIENCE,
        "arima_order": list(ARIMA_ORDER),
        "glsar_rho": GLSAR_RHO,
        "glsar_maxiter": GLSAR_MAXITER,
        "ensemble_history": MIN_ENSEMBLE_HISTORY,
        "ensemble_eps": ENSEMBLE_EPS,
        "model_names": MODEL_NAMES,
        "model_features": MODEL_FEATURES,
        "vs_lstm_features": VS_LSTM_FEATURES,
        "target_rule": "y[t] = 100*log(P[t+h]/P[t])",
        "first_valid_origin_rule": "INITIAL_TRAIN + horizon - 1",
        "training_target_end_rule": "origin - horizon + 1",
        "checkpoint_schema": "one_row_per_origin_per_model",
    }


def _load_partial_checkpoint(
    hname: str,
    h: int,
    df: pd.DataFrame,
    y: np.ndarray,
    dataset_sha256: str
) -> Tuple[List[Dict], Dict[str, List[float]], Optional[int]]:
    """
    Load a partial checkpoint and reconstruct causal ensemble error history.

    The checkpoint stores one row per model/origin. The last completed origin
    is therefore the maximum origin_index present in the file.
    """
    cp = checkpoint_path(hname)
    meta_path = CHECKPOINT_DIR / f"checkpoint_meta_{hname}.json"
    expected_cfg = checkpoint_config(hname, h, dataset_sha256)

    if not cp.exists() or not meta_path.exists():
        return [], {m: [] for m in MODEL_NAMES}, None

    try:
        saved_cfg = json.loads(meta_path.read_text(encoding="utf-8"))
        if saved_cfg != expected_cfg:
            log(f"{hname}: checkpoint configuration mismatch; starting fresh.")
            return [], {m: [] for m in MODEL_NAMES}, None
        old = pd.read_csv(cp)
    except Exception as exc:
        log(f"{hname}: checkpoint read failed; starting fresh: {exc}")
        return [], {m: [] for m in MODEL_NAMES}, None

    if old.empty:
        return [], {m: [] for m in MODEL_NAMES}, None

    required = {
        "origin_index", "model", "actual_return", "predicted_return"
    }
    if not required.issubset(old.columns):
        log(f"{hname}: incompatible checkpoint; starting fresh.")
        return [], {m: [] for m in MODEL_NAMES}, None

    # Validate that the stored target date/index relationship is consistent
    # with the current horizon before resuming.
    if "target_date" in old.columns and "origin_date" in old.columns:
        try:
            dates = pd.to_datetime(df["date"]).reset_index(drop=True)
            sample = old[["origin_index", "target_date"]].drop_duplicates("origin_index")
            for _, r in sample.iterrows():
                oi = int(r["origin_index"])
                if oi < 0 or oi + h >= len(df):
                    raise ValueError("origin outside current dataset")
                expected_target = str(dates.iloc[oi + h].date())
                if str(r["target_date"])[:10] != expected_target:
                    raise ValueError(
                        f"target-date mismatch at origin {oi}: "
                        f"stored={r['target_date']}, expected={expected_target}"
                    )
        except Exception as exc:
            log(f"{hname}: checkpoint target-alignment validation failed; starting fresh: {exc}")
            return [], {m: [] for m in MODEL_NAMES}, None

    old = old.sort_values(["origin_index", "model"]).copy()

    # Strict schema validation: exactly one row for every expected model at
    # each origin. A mere count of unique origins is insufficient because a
    # corrupted checkpoint could contain one model repeated at every origin.
    expected_models = set(MODEL_NAMES)
    model_counts = old.groupby("origin_index")["model"].nunique()
    complete_origins = []

    for origin, g in old.groupby("origin_index", sort=True):
        models_here = set(g["model"].astype(str))
        row_count = len(g)
        if row_count == len(MODEL_NAMES) and models_here == expected_models:
            complete_origins.append(int(origin))

    if not complete_origins:
        log(f"{hname}: no complete origin found in checkpoint; starting fresh.")
        return [], {m: [] for m in MODEL_NAMES}, None

    complete_origins = np.asarray(sorted(complete_origins), dtype=int)
    last_complete = int(complete_origins.max())

    # Keep only the contiguous prefix of complete origins beginning at the
    # expected first origin. This prevents a later orphaned origin from being
    # mistaken for a valid resumable prefix.
    expected_origins = valid_forecast_origins(
        len(df), h, MAX_FORECASTS
    )
    prefix = expected_origins[expected_origins <= last_complete]
    if len(prefix) == 0:
        return [], {m: [] for m in MODEL_NAMES}, None

    complete_set = set(complete_origins.tolist())
    contiguous = []
    for o in prefix:
        if int(o) not in complete_set:
            break
        contiguous.append(int(o))

    if not contiguous:
        return [], {m: [] for m in MODEL_NAMES}, None

    last_complete = contiguous[-1]
    old = old[old["origin_index"].isin(contiguous)].copy()

    # Reconstruct the exact previous OOS error history needed by the
    # adaptive ensemble. This preserves causality after a restart.
    history_errors = {m: [] for m in MODEL_NAMES}
    for origin in sorted(old["origin_index"].unique()):
        g = old[old["origin_index"] == origin]
        for _, r in g.iterrows():
            model = str(r["model"])
            if model in history_errors:
                history_errors[model].append(
                    float(r["predicted_return"] - r["actual_return"])
                )

    log(
        f"{hname}: partial checkpoint loaded: "
        f"{len(old)} rows, last complete origin={last_complete}"
    )

    return old.to_dict("records"), history_errors, last_complete


def _write_partial_checkpoint(
    hname: str,
    result_df: pd.DataFrame,
    comp_row: Optional[Dict] = None,
    config: Optional[Dict] = None
) -> None:
    """
    Atomic-ish checkpoint write: write to a temporary file then replace.
    """
    cp = checkpoint_path(hname)
    tmp = cp.with_suffix(".tmp.csv")

    result_df.to_csv(
        tmp,
        index=False,
        encoding="utf-8-sig"
    )
    tmp.replace(cp)

    if config is not None:
        meta_path = CHECKPOINT_DIR / f"checkpoint_meta_{hname}.json"
        meta_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    if comp_row is not None:
        cpath = CHECKPOINT_DIR / f"computational_checkpoint_{hname}.csv"
        pd.DataFrame([comp_row]).to_csv(
            cpath,
            index=False,
            encoding="utf-8-sig"
        )


def run_horizon(
    df: pd.DataFrame,
    h_name: str,
    h: int,
    dataset_sha256: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    y = direct_target(df, h)
    origins = valid_forecast_origins(len(df), h, MAX_FORECASTS)

    if len(origins) == 0:
        return pd.DataFrame(), pd.DataFrame()

    log(
        f"START {h_name}: h={h}, valid origins={len(origins)}, "
        f"first={origins[0]}, last={origins[-1]}"
    )

    # Resume from a partial checkpoint if present.
    records, history_errors, last_complete = _load_partial_checkpoint(
        h_name, h, df, y, dataset_sha256
    )

    if last_complete is not None:
        origins = origins[origins > last_complete]
        log(
            f"{h_name}: resuming with {len(origins)} remaining origins."
        )

    # If the checkpoint already contains all valid origins, simply return it.
    if len(origins) == 0:
        result_df = pd.DataFrame(records)
        if len(result_df):
            result_df["regime"] = assign_training_only_regimes(
                df, result_df, h
            )
        comp_path = CHECKPOINT_DIR / f"computational_checkpoint_{h_name}.csv"
        if comp_path.exists():
            comp_df = pd.read_csv(comp_path)
        else:
            comp_df = pd.DataFrame()
        return result_df, comp_df

    fitted = None
    current_fit_origin = None
    cumulative_fit_seconds = {m: 0.0 for m in FITTABLE_MODELS}
    fit_count = 0
    prior_session_fit_count = 0
    resumed_from_checkpoint = last_complete is not None

    # Restore cumulative computational audit values when available so a
    # resumed run does not report only the post-restart session.
    comp_path = CHECKPOINT_DIR / f"computational_checkpoint_{h_name}.csv"
    if resumed_from_checkpoint and comp_path.exists():
        try:
            prev_comp = pd.read_csv(comp_path)
            if not prev_comp.empty:
                row = prev_comp.iloc[-1]
                prior_session_fit_count = int(row.get("fit_count_total", row.get("fit_count", 0)))
                fit_count = prior_session_fit_count
                for m in FITTABLE_MODELS:
                    key = f"{m}_fit_sec"
                    if key in row.index and pd.notna(row[key]):
                        cumulative_fit_seconds[m] = float(row[key])
        except Exception as exc:
            log(f"{h_name}: computational checkpoint restore failed; starting audit counters fresh: {exc}")

    start_horizon = time.perf_counter()

    for k, origin in enumerate(origins, start=1):

        if current_fit_origin is None or (
            origin - current_fit_origin >= RETRAIN_EVERY
        ):
            t_fit = time.perf_counter()

            fitted, audit = fit_models(df, y, origin, h)

            fit_elapsed = time.perf_counter() - t_fit
            fit_count += 1
            current_fit_origin = origin

            for key in cumulative_fit_seconds:
                cumulative_fit_seconds[key] += float(
                    audit.get(f"{key}_sec", 0.0)
                )

            log(
                f"{h_name}: retrain {fit_count} at origin {origin}; "
                f"fit_elapsed={fit_elapsed:.2f}s"
            )

            # Save a checkpoint immediately after a successful retraining
            # as an additional crash-recovery point.
            if SAVE_AFTER_EVERY_RETRAIN and records:
                comp_row = {
                    "horizon": h_name,
                    "horizon_days": h,
                    "forecast_origins_completed": len(
                        set(r["origin_index"] for r in records)
                    ),
                    "fit_count_total": fit_count,
                    "fit_count_current_session": fit_count - prior_session_fit_count,
                    "ARIMA_fit_sec_current_session":
                        cumulative_fit_seconds["ARIMA"],
                    "OLS_fit_sec_current_session":
                        cumulative_fit_seconds["OLS"],
                    "GLS_fit_sec_current_session":
                        cumulative_fit_seconds["GLS"],
                    "LSTM_fit_sec_current_session":
                        cumulative_fit_seconds["LSTM"],
                    "VS_LSTM_fit_sec_current_session":
                        cumulative_fit_seconds["VS-LSTM"],
                    "GLS_LSTM_fit_sec_current_session":
                        cumulative_fit_seconds["GLS-LSTM"],
                }
                _write_partial_checkpoint(
                    h_name, pd.DataFrame(records), comp_row,
                    checkpoint_config(h_name, h, dataset_sha256)
                )

        pred = predict_fitted_models(
            fitted, df, y, origin, h, history_errors
        )

        actual = float(y[origin])
        current_price = float(df.iloc[origin]["price"])
        future_price = float(df.iloc[origin + h]["price"])

        ensemble_info = pred.get("__ensemble_info__", "{}")
        for model_name, p in pred.items():
            if model_name.startswith("__"):
                continue
            p = float(p)

            pred_price = current_price * math.exp(
                np.clip(p / 100.0, -50, 50)
            )

            records.append({
                "horizon": h_name,
                "horizon_days": h,
                "origin_index": int(origin),
                "origin_date": str(df.iloc[origin]["date"].date()),
                "target_date": str(df.iloc[origin + h]["date"].date()),
                "model": model_name,
                "actual_return": actual,
                "predicted_return": p,
                "current_price": current_price,
                "actual_future_price": future_price,
                "predicted_future_price": pred_price,
                "actual_sign": int(np.sign(actual)),
                "predicted_sign": int(np.sign(p)),
                "ensemble_members_weights": ensemble_info if model_name == "Adaptive Ensemble" else "",
                "regime": "Unknown",
            })

            # Current OOS error is appended only after prediction/observation.
            history_errors.setdefault(model_name, []).append(p - actual)

        elapsed = time.perf_counter() - start_horizon

        if k % 25 == 0 or k == len(origins):
            log(
                f"{h_name}: {k}/{len(origins)} forecasts complete; "
                f"elapsed={elapsed/60:.2f} min"
            )

        # V6.2: periodic checkpointing prevents loss of hours of computation.
        if (k % CHECKPOINT_EVERY == 0) or (k == len(origins)):
            partial = pd.DataFrame(records)
            _write_partial_checkpoint(
                h_name, partial,
                config=checkpoint_config(h_name, h, dataset_sha256)
            )
            log(
                f"{h_name}: checkpoint saved after {k}/{len(origins)} "
                f"new forecasts."
            )

        if k % 25 == 0:
            gc.collect()

    result_df = pd.DataFrame(records)

    result_df["regime"] = assign_training_only_regimes(
        df, result_df, h
    )

    total_runtime = time.perf_counter() - start_horizon
    comp_rows = [{
        "horizon": h_name,
        "horizon_days": h,
        "forecast_origins": len(
            result_df["origin_index"].unique()
        ),
        "fit_count": fit_count,
        "fit_count_current_session": fit_count - prior_session_fit_count,
        "resumed_from_checkpoint": resumed_from_checkpoint,
        "total_horizon_runtime_sec": total_runtime,
        "ARIMA_fit_sec": cumulative_fit_seconds["ARIMA"],
        "OLS_fit_sec": cumulative_fit_seconds["OLS"],
        "GLS_fit_sec": cumulative_fit_seconds["GLS"],
        "LSTM_fit_sec": cumulative_fit_seconds["LSTM"],
        "VS_LSTM_fit_sec": cumulative_fit_seconds["VS-LSTM"],
        "GLS_LSTM_fit_sec": cumulative_fit_seconds["GLS-LSTM"],
        "V6_2_1_resume_enabled": True,
        "V6_2_1_checkpoint_every": CHECKPOINT_EVERY,
    }]

    comp_df = pd.DataFrame(comp_rows)

    # Final checkpoint for the completed horizon.
    _write_partial_checkpoint(
        h_name, result_df, comp_rows[0],
        checkpoint_config(h_name, h, dataset_sha256)
    )

    return result_df, comp_df


# =============================================================================
# 15. TRAINING-ONLY REGIME CLASSIFICATION
# =============================================================================

def assign_training_only_regimes(
    df: pd.DataFrame,
    result_df: pd.DataFrame,
    h: int
) -> pd.Series:

    regimes = []

    for _, r in result_df.iterrows():
        origin = int(r["origin_index"])

        # Use only data strictly before the origin.
        train_end = origin
        vol = (
            df.iloc[:train_end]["return_log_pct"]
            .rolling(ROLLING_VOL_WINDOW)
            .std()
            .dropna()
        )

        if len(vol) < 60:
            regimes.append("Unknown")
            continue

        q1 = float(vol.quantile(1/3))
        q2 = float(vol.quantile(2/3))
        current_vol = float(vol.iloc[-1])

        if current_vol <= q1:
            regimes.append("Low Volatility")
        elif current_vol >= q2:
            regimes.append("High Volatility")
        else:
            # Direction is determined only from the historical return sign.
            recent_ret = float(
                df.iloc[max(0, train_end - 30):train_end]["return_log_pct"].sum()
            )
            regimes.append("Bull" if recent_ret >= 0 else "Bear")

    return pd.Series(regimes, index=result_df.index)


# =============================================================================
# 16. METRICS
# =============================================================================

def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if n == 0:
        return np.nan, np.nan

    phat = k / n
    den = 1 + z**2 / n
    center = (phat + z**2 / (2*n)) / den
    half = (
        z / den
        * math.sqrt(
            phat*(1-phat)/n + z**2/(4*n**2)
        )
    )
    return max(0.0, center-half), min(1.0, center+half)


def exact_mcnemar_pvalue(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value conditional on discordant pairs."""
    n = b + c
    if n == 0:
        return np.nan
    return float(min(1.0, 2.0 * stats.binom.cdf(min(b, c), n, 0.5)))


def compute_metrics(result_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (horizon, hdays, model), g in result_df.groupby(["horizon", "horizon_days", "model"]):
        a = g["actual_return"].to_numpy(float)
        p = g["predicted_return"].to_numpy(float)
        e = p - a
        rmse = float(np.sqrt(np.mean(e**2)))
        mae = float(np.mean(np.abs(e)))

        actual_sign = np.sign(a)
        pred_sign = np.sign(p)
        mask = (actual_sign != 0) & (pred_sign != 0)
        n_dir = int(mask.sum())
        da = float(np.mean(actual_sign[mask] == pred_sign[mask]) * 100) if n_dir else np.nan
        coverage = float(mask.mean() * 100)

        ba = mcc = f1 = always_up = excess_up = mcnemar_p = np.nan
        if n_dir:
            yt = (actual_sign[mask] > 0).astype(int)
            yp = (pred_sign[mask] > 0).astype(int)
            tp = int(np.sum((yt == 1) & (yp == 1)))
            tn = int(np.sum((yt == 0) & (yp == 0)))
            fp = int(np.sum((yt == 0) & (yp == 1)))
            fn = int(np.sum((yt == 1) & (yp == 0)))
            tpr = tp / (tp + fn) if tp + fn else np.nan
            tnr = tn / (tn + fp) if tn + fp else np.nan
            ba = float((tpr + tnr) / 2) if np.isfinite(tpr) and np.isfinite(tnr) else np.nan
            den = math.sqrt(max((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn), 0))
            if den > 0:
                mcc = float((tp*tn - fp*fn) / den)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0
            always_up = 100.0 * np.mean(yt == 1)
            excess_up = da - always_up
            model_correct = yp == yt
            up_correct = yt == 1
            b = int(np.sum(model_correct & ~up_correct))
            c = int(np.sum(~model_correct & up_correct))
            mcnemar_p = exact_mcnemar_pvalue(b, c)

        ap = g["actual_future_price"].to_numpy(float)
        pp = g["predicted_future_price"].to_numpy(float)
        pe = pp - ap
        rows.append({
            "horizon": horizon, "horizon_days": hdays, "model": model, "n": len(g),
            "RMSE_return": rmse, "MAE_return": mae,
            "Directional_Accuracy_pct": da, "Balanced_Accuracy_pct": 100 * ba if np.isfinite(ba) else np.nan,
            "F1_positive": f1, "MCC_direction": mcc, "DA_coverage_pct": coverage,
            "AlwaysUp_accuracy_same_coverage_pct": always_up,
            "DA_excess_vs_AlwaysUp_pp": excess_up,
            "McNemar_exact_p_vs_AlwaysUp": mcnemar_p,
            "Price_RMSE": float(np.sqrt(np.mean(pe**2))),
            "Price_MAE": float(np.mean(np.abs(pe))),
            "Price_MAPE_pct": float(np.mean(np.abs(pe) / np.maximum(np.abs(ap), EPS)) * 100),
        })
    return pd.DataFrame(rows)


# =============================================================================
# 17. BASE-RATE / 100%-POSITIVE AUDIT
# =============================================================================

def sign_distribution_analysis(result_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for hname, g in result_df.groupby("horizon"):
        actual = g.drop_duplicates(
            subset=["origin_index"]
        )["actual_return"].to_numpy(float)

        n = len(actual)
        positive_n = int(np.sum(actual > 0))
        negative_n = int(np.sum(actual < 0))
        zero_n = int(np.sum(actual == 0))

        pval = stats.binomtest(
            positive_n,
            n=n,
            p=0.5,
            alternative="two-sided"
        ).pvalue if n else np.nan

        ci_low, ci_high = wilson_interval(positive_n, n)

        rows.append({
            "horizon": hname,
            "n_origins": n,
            "positive_n": positive_n,
            "negative_n": negative_n,
            "zero_n": zero_n,
            "positive_pct": 100 * positive_n / n if n else np.nan,
            "positive_rate_CI95_low_pct": 100 * ci_low,
            "positive_rate_CI95_high_pct": 100 * ci_high,
            "exact_binomial_p_vs_0.5": pval,
            "all_positive": bool(positive_n == n and n > 0),
            "interpretation": (
                "High unconditional positive-return rate; not by itself "
                "evidence of conditional predictability."
            ),
        })

    return pd.DataFrame(rows)


def calendar_positive_rate_analysis(
    df: pd.DataFrame,
    result_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Divide forecast origins into calendar-year blocks to determine whether
    long-horizon positive-rate concentration is localized in time.
    """
    r = result_df.drop_duplicates(
        subset=["horizon", "origin_index"]
    ).copy()
    r["year"] = pd.to_datetime(r["origin_date"]).dt.year

    rows = []

    for (horizon, year), g in r.groupby(["horizon", "year"]):
        a = g["actual_return"].to_numpy(float)
        n = len(a)
        k = int(np.sum(a > 0))
        lo, hi = wilson_interval(k, n)

        rows.append({
            "horizon": horizon,
            "calendar_year": int(year),
            "n": n,
            "positive_n": k,
            "negative_n": int(np.sum(a < 0)),
            "positive_pct": 100*k/n if n else np.nan,
            "CI95_low_pct": 100*lo,
            "CI95_high_pct": 100*hi,
        })

    return pd.DataFrame(rows)


# =============================================================================
# 18. SKILL SCORES
# =============================================================================

def add_skill_scores(metrics: pd.DataFrame) -> pd.DataFrame:
    out = metrics.copy()

    for horizon, g in out.groupby("horizon"):
        mean_rmse = float(
            g.loc[g["model"] == "Historical Mean", "RMSE_return"].iloc[0]
        )
        zero_rmse = float(
            g.loc[g["model"] == "Zero Return", "RMSE_return"].iloc[0]
        )

        idx = out["horizon"] == horizon

        out.loc[idx, "Skill_vs_HistoricalMean"] = (
            1.0 - out.loc[idx, "RMSE_return"]**2 / max(mean_rmse**2, EPS)
        )

        out.loc[idx, "Skill_vs_ZeroReturn"] = (
            1.0 - out.loc[idx, "RMSE_return"]**2 / max(zero_rmse**2, EPS)
        )

    return out


# =============================================================================
# 19. PAIRWISE HAC DM
# =============================================================================

def newey_west_long_run_variance(x: np.ndarray, bandwidth: int) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) < 3:
        return np.nan

    x = x - np.mean(x)
    n = len(x)

    gamma0 = np.dot(x, x) / n
    lr = gamma0

    for lag in range(1, min(bandwidth, n - 1) + 1):
        gamma = np.dot(x[lag:], x[:-lag]) / n
        weight = 1.0 - lag / (bandwidth + 1.0)
        lr += 2.0 * weight * gamma

    return float(max(lr, 0.0))


def dm_test_hac(
    e1: np.ndarray,
    e2: np.ndarray,
    h: int
) -> Tuple[float, float, int]:
    """
    Squared-error loss differential.

    Bandwidth is overlap-aware: at least h-1 for direct h-step cumulative
    overlapping returns, with an additional finite-sample floor.
    """
    d = np.asarray(e1)**2 - np.asarray(e2)**2
    d = d[np.isfinite(d)]

    n = len(d)
    if n < 10:
        return np.nan, np.nan, 0

    bandwidth = min(
        n - 1,
        max(h - 1, int(round(n ** (1/3))))
    )

    lrv = newey_west_long_run_variance(d, bandwidth)
    if not np.isfinite(lrv) or lrv <= EPS:
        return np.nan, np.nan, bandwidth

    mean_d = float(np.mean(d))
    se = math.sqrt(lrv / n)
    dm = mean_d / max(se, EPS)

    p = 2.0 * (1.0 - stats.norm.cdf(abs(dm)))
    return float(dm), float(p), int(bandwidth)


def dm_pairwise(result_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (hname, h), g in result_df.groupby(["horizon", "horizon_days"]):
        pivot = g.pivot(
            index="origin_index",
            columns="model",
            values=["actual_return", "predicted_return"]
        )

        models = sorted(
            g["model"].unique()
        )

        for i, m1 in enumerate(models):
            for m2 in models[i+1:]:
                if ("actual_return", m1) not in pivot.columns:
                    continue
                if ("predicted_return", m1) not in pivot.columns:
                    continue

                a1 = pivot[("actual_return", m1)].to_numpy(float)
                p1 = pivot[("predicted_return", m1)].to_numpy(float)
                a2 = pivot[("actual_return", m2)].to_numpy(float)
                p2 = pivot[("predicted_return", m2)].to_numpy(float)

                mask = np.isfinite(a1+p1+a2+p2)
                e1 = p1[mask] - a1[mask]
                e2 = p2[mask] - a2[mask]

                dm, p, bw = dm_test_hac(e1, e2, h)

                rows.append({
                    "horizon": hname,
                    "horizon_days": h,
                    "model_1": m1,
                    "model_2": m2,
                    "n": len(e1),
                    "DM_HAC": dm,
                    "p_value_HAC": p,
                    "HAC_bandwidth": bw,
                    "overlap_warning": bool(h - 1 >= len(e1)),
                })

    return pd.DataFrame(rows)


def dm_best_nonensemble(result_df: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    """One-sided HAC-DM test where H1 is that Adaptive Ensemble has lower loss."""
    rows=[]
    for (hname,h), g in result_df.groupby(["horizon","horizon_days"]):
        mg=metrics[(metrics.horizon==hname)&(metrics.horizon_days==h)]
        cand=mg[mg.model!="Adaptive Ensemble"].dropna(subset=["RMSE_return"])
        if cand.empty or not (mg.model=="Adaptive Ensemble").any():
            continue
        best=cand.sort_values("RMSE_return").iloc[0]
        bm=str(best.model)
        pivot=g.pivot(index="origin_index",columns="model",values="actual_return")
        pred=g.pivot(index="origin_index",columns="model",values="predicted_return")
        if "Adaptive Ensemble" not in pred or bm not in pred:
            continue
        a=pivot["Adaptive Ensemble"].to_numpy(float)
        ea=pred["Adaptive Ensemble"].to_numpy(float)-a
        eb=pred[bm].to_numpy(float)-a
        mask=np.isfinite(ea)&np.isfinite(eb)
        dm,p2,bw=dm_test_hac(ea[mask],eb[mask],h)
        p1=float(stats.norm.cdf(dm)) if np.isfinite(dm) else np.nan
        ae_rmse=float(mg.loc[mg.model=="Adaptive Ensemble","RMSE_return"].iloc[0])
        best_rmse=float(best.RMSE_return)
        rows.append({"horizon":hname,"horizon_days":h,"best_nonensemble_model":bm,
                     "AE_RMSE":ae_rmse,"best_nonensemble_RMSE":best_rmse,
                     "RMSE_improvement_pct":100*(best_rmse-ae_rmse)/best_rmse,
                     "n":int(mask.sum()),"DM_HAC":dm,"p_value_two_sided":p2,
                     "p_value_one_sided_AE_better":p1,"HAC_bandwidth":bw,
                     "decision_alpha_0.05":bool(np.isfinite(p1) and p1<0.05)})
    return pd.DataFrame(rows)


# =============================================================================
# 20. MCS
# =============================================================================

def moving_block_sample(
    x: np.ndarray,
    block_length: int,
    rng: np.random.Generator
) -> np.ndarray:
    n = len(x)
    if n == 0:
        return x

    block_length = max(1, min(block_length, n))
    starts = rng.integers(0, max(1, n - block_length + 1), size=math.ceil(n / block_length))
    blocks = [x[s:s+block_length] for s in starts]
    return np.concatenate(blocks)[:n]


def bootstrap_mcs_elimination(loss_matrix: np.ndarray, model_names: List[str], alpha: float,
                               B: int, block_length: int, seed: int) -> Tuple[List[str], float]:
    """Bootstrap MCS approximation using a Hansen-style range/TR statistic.

    It is intentionally reported as an approximation rather than an exact
    reproduction of the finite-sample Hansen--Lunde--Nason algorithm.
    """
    rng=np.random.default_rng(seed)
    active=list(range(loss_matrix.shape[1]))
    p_last=np.nan
    while len(active)>1:
        L=loss_matrix[:,active]
        n=L.shape[0]
        d=L-np.mean(L,axis=1,keepdims=True)
        dbar=np.mean(d,axis=0)
        boot=np.empty((B,len(active)),float)
        for b in range(B):
            idx=moving_block_indices(n,block_length,rng)
            boot[b]=np.mean(d[idx],axis=0)
        se=np.std(boot,axis=0,ddof=1)
        t=np.divide(dbar,se,out=np.zeros_like(dbar),where=se>0)
        tr=float(np.max(t))
        bt=np.empty(B,float)
        for b in range(B):
            z=boot[b]
            bt[b]=float(np.max(np.divide(z-np.mean(z),se,out=np.zeros_like(z),where=se>0)))
        p_last=float(np.mean(bt>=tr))
        if p_last>=alpha:
            break
        active.pop(int(np.argmax(t)))
    return [model_names[i] for i in active],p_last


def mcs_sensitivity(result_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (hname, h), g in result_df.groupby(["horizon", "horizon_days"]):
        pivot_actual = g.pivot(
            index="origin_index",
            columns="model",
            values="actual_return"
        )
        pivot_pred = g.pivot(
            index="origin_index",
            columns="model",
            values="predicted_return"
        )

        common_models = [
            m for m in pivot_pred.columns
            if m in pivot_actual.columns
        ]

        # Squared error loss.
        loss = (pivot_pred[common_models] - pivot_actual[common_models])**2
        loss = loss.dropna(how="any")

        if len(loss) < 20:
            continue

        L = loss.to_numpy(float)

        for alpha in MCS_ALPHAS:
            members, mcs_p = bootstrap_mcs_elimination(
                L, common_models, alpha, MCS_BOOTSTRAPS,
                min(MCS_BLOCK_LENGTH, len(L)), RANDOM_SEED + int(h)
            )

            rows.append({
                "horizon": hname,
                "horizon_days": h,
                "alpha": alpha,
                "n": len(loss),
                "MCS_members": ", ".join(members),
                "MCS_size": len(members),
                "MCS_p_value": mcs_p,
                "MCS_method": "bootstrap Hansen-style TR approximation",
            })

    return pd.DataFrame(rows)


# =============================================================================
# 21. NON-OVERLAP + MOVING-BLOCK BOOTSTRAP ROBUSTNESS
# =============================================================================

def robustness_analysis(result_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    rng_master = np.random.default_rng(RANDOM_SEED)

    for (hname, h, model), g in result_df.groupby(
        ["horizon", "horizon_days", "model"]
    ):
        g = g.sort_values("origin_index").reset_index(drop=True)

        # Non-overlap subset: every h-th forecast origin.
        nonoverlap = g.iloc[::max(1, h)].copy()

        # Recent half of the sample.
        recent = g.iloc[len(g)//2:].copy()

        for label, sub in [
            ("nonoverlap", nonoverlap),
            ("recent", recent),
        ]:
            a = sub["actual_return"].to_numpy(float)
            p = sub["predicted_return"].to_numpy(float)

            if len(a) == 0:
                continue

            e = p - a
            rmse = float(np.sqrt(np.mean(e**2)))
            mae = float(np.mean(np.abs(e)))

            # Bootstrap CI for RMSE using moving blocks.
            boot_rmses = []

            # Long-horizon bootstrap can be expensive; use the full requested
            # 10,000 iterations only when sample size permits. Otherwise the
            # exact requested number is still used but blocks are recycled.
            for _ in range(ROBUSTNESS_BOOTSTRAPS):
                idx = moving_block_indices(
                    len(e),
                    min(
                        max(1, h),
                        max(1, len(e))
                    ),
                    rng_master
                )
                eb = e[idx]
                boot_rmses.append(
                    math.sqrt(np.mean(eb**2))
                )

            lo, hi = np.quantile(boot_rmses, [0.025, 0.975])

            rows.append({
                "horizon": hname,
                "horizon_days": h,
                "model": model,
                "sample_type": label,
                "n": len(sub),
                "RMSE": rmse,
                "MAE": mae,
                "RMSE_bootstrap_CI95_low": float(lo),
                "RMSE_bootstrap_CI95_high": float(hi),
                "bootstrap_B": ROBUSTNESS_BOOTSTRAPS,
                "block_length": min(max(1, h), max(1, len(e))),
                "interpretation_note": (
                    "Diagnostic robustness only; non-overlap samples at long "
                    "horizons have limited effective size."
                ),
            })

    return pd.DataFrame(rows)


def moving_block_indices(
    n: int,
    block_length: int,
    rng: np.random.Generator
) -> np.ndarray:
    if n <= 1:
        return np.zeros(n, dtype=int)

    block_length = max(1, min(block_length, n))
    out = []

    while len(out) < n:
        start = int(rng.integers(0, n - block_length + 1))
        out.extend(range(start, start + block_length))

    return np.asarray(out[:n], dtype=int)


def stationarity_break_diagnostics(df: pd.DataFrame, results: pd.DataFrame) -> Tuple[pd.DataFrame,pd.DataFrame]:
    """ADF/KPSS diagnostics and CUSUM mean-stability diagnostics."""
    srows=[]; brows=[]
    from statsmodels.tsa.stattools import adfuller, kpss
    for hname,h in HORIZONS.items():
        y=direct_target(df,h); y=y[np.isfinite(y)]
        try:
            adf=adfuller(y,autolag="AIC",regression="c")
            adf_stat,adf_p,adf_lag,adf_n,adf_crit=adf[0],adf[1],adf[2],adf[3],adf[4]
        except Exception:
            adf_stat=adf_p=adf_lag=adf_n=np.nan; adf_crit={}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kp=kpss(y,regression="c",nlags="auto")
            kp_stat,kp_p,kp_lag,kp_crit=kp[0],kp[1],kp[2],kp[3]
        except Exception:
            kp_stat=kp_p=kp_lag=np.nan; kp_crit={}
        srows.append({"horizon":hname,"horizon_days":h,"n":len(y),"ADF_stat":adf_stat,"ADF_p":adf_p,
                      "ADF_1pct":adf_crit.get("1%",np.nan),"ADF_5pct":adf_crit.get("5%",np.nan),"ADF_10pct":adf_crit.get("10%",np.nan),
                      "KPSS_stat":kp_stat,"KPSS_p":kp_p,"KPSS_1pct":kp_crit.get("1%",np.nan),
                      "KPSS_5pct":kp_crit.get("5%",np.nan),"KPSS_10pct":kp_crit.get("10%",np.nan)})
        rg=results[(results.horizon==hname)&(results.model=="Adaptive Ensemble")].sort_values("origin_index")
        for series_name,series in [("target_return",y),("adaptive_ensemble_error",(rg.predicted_return-rg.actual_return).to_numpy(float))]:
            z=np.asarray(series,float); z=z[np.isfinite(z)]
            if len(z)<50: continue
            try:
                stat,p,crit=breaks_cusumolsresid(z-np.mean(z),ddof=1)
            except Exception:
                stat=p=np.nan
            brows.append({"horizon":hname,"horizon_days":h,"series":series_name,"n":len(z),"CUSUM_stat":stat,"CUSUM_p":p,
                          "interpretation":"Mean-stability diagnostic; no breakpoint date is inferred."})
    return pd.DataFrame(srows),pd.DataFrame(brows)


# =============================================================================
# 22. COMPUTATIONAL AUDIT
# =============================================================================

def computational_audit(
    comp_df: pd.DataFrame
) -> pd.DataFrame:
    out = comp_df.copy()

    out["python_version"] = sys.version
    out["platform"] = platform.platform()
    out["processor"] = platform.processor()
    out["machine"] = platform.machine()
    out["numpy_version"] = np.__version__
    out["pandas_version"] = pd.__version__
    out["scipy_version"] = scipy.__version__
    out["statsmodels_version"] = sm.__version__
    out["tensorflow_available"] = TF_AVAILABLE

    if TF_AVAILABLE:
        out["tensorflow_version"] = tf.__version__
        try:
            gpus = tf.config.list_physical_devices("GPU")
            out["GPU_count"] = len(gpus)
            out["GPU_devices"] = str(gpus)
        except Exception:
            out["GPU_count"] = np.nan
            out["GPU_devices"] = ""
    else:
        out["tensorflow_version"] = ""
        out["GPU_count"] = 0
        out["GPU_devices"] = ""

    out["random_seed"] = RANDOM_SEED
    out["initial_train"] = INITIAL_TRAIN
    out["retrain_every"] = RETRAIN_EVERY
    out["LSTM_epochs_max"] = LSTM_EPOCHS
    out["LSTM_batch_size"] = LSTM_BATCH_SIZE
    out["LSTM_early_stopping_patience"] = LSTM_PATIENCE
    out["GLSAR_maxiter"] = GLSAR_MAXITER
    out["ARIMA_order"] = str(ARIMA_ORDER)
    out["MCS_B"] = MCS_BOOTSTRAPS
    out["robustness_bootstrap_B"] = ROBUSTNESS_BOOTSTRAPS
    out["notes"] = (
        "Primary full run uses one prespecified seed; multi-seed sensitivity "
        "is reported separately to avoid multiplying the main full-run cost."
    )

    return out


# =============================================================================
# 23. ABLATION STUDY
# =============================================================================

def ensemble_ablation(metrics: pd.DataFrame) -> pd.DataFrame:
    """
    This V6 component is a reporting-oriented ablation based on already
    generated forecast-level predictions.

    It compares the adaptive ensemble with:
      - classical-only equal mean
      - neural-only equal mean
      - hybrid-only equal mean

    Because the adaptive ensemble is causal but these equal-weight composites
    are post-hoc diagnostics, they are explicitly labelled ablations.
    """
    rows = []

    groups = {
        "ClassicalOnly": [
            "Zero Return",
            "Historical Mean",
            "ARIMA",
            "OLS",
            "GLS",
        ],
        "NeuralOnly": [
            "LSTM",
            "VS-LSTM",
        ],
        "HybridOnly": [
            "GLS-LSTM",
            "Adaptive Ensemble",
        ],
    }

    for (hname, h), g in metrics.groupby(["horizon", "horizon_days"]):
        for label, members in groups.items():
            sub = g[g["model"].isin(members)]

            if len(sub) == 0:
                continue

            rows.append({
                "horizon": hname,
                "horizon_days": h,
                "ablation_group": label,
                "available_models": ", ".join(sub["model"].tolist()),
                "mean_RMSE_of_members": float(sub["RMSE_return"].mean()),
                "best_member_RMSE": float(sub["RMSE_return"].min()),
            })

    return pd.DataFrame(rows)


# =============================================================================
# 24. MULTI-SEED SENSITIVITY
# =============================================================================

def run_multi_seed_sensitivity(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Computationally bounded sensitivity:
    selected horizons, first MULTI_SEED_MAX_ORIGINS origins.

    This does NOT replace the primary full-data run.
    """
    if not RUN_MULTI_SEED_SENSITIVITY or not TF_AVAILABLE:
        return pd.DataFrame()

    rows = []

    for hname in MULTI_SEED_HORIZONS:
        h = HORIZONS[hname]
        origins = valid_forecast_origins(
            len(df), h, MULTI_SEED_MAX_ORIGINS
        )

        if len(origins) == 0:
            continue

        y = direct_target(df, h)

        for seed in MULTI_SEEDS:
            set_global_seed(seed)

            for origin in origins:
                train_target_end = origin - h + 1
                if np.sum(np.isfinite(y[:train_target_end])) < INITIAL_TRAIN:
                    continue

                t0 = time.perf_counter()
                fitted = fit_lstm(
                    df, y, train_target_end, seed
                )
                fit_sec = time.perf_counter() - t0

                p = predict_lstm(
                    fitted, df, origin
                )
                a = float(y[origin])

                rows.append({
                    "horizon": hname,
                    "horizon_days": h,
                    "seed": seed,
                    "origin_index": int(origin),
                    "actual_return": a,
                    "predicted_return": p,
                    "squared_error": (p-a)**2,
                    "absolute_error": abs(p-a),
                    "fit_runtime_sec": fit_sec,
                })

    return pd.DataFrame(rows)


# =============================================================================
# 25. FIGURES
# =============================================================================

def make_figures(metrics: pd.DataFrame) -> None:
    for metric, ylabel, fname in [
        ("RMSE_return", "Return RMSE", "RMSE_by_horizon_V6.png"),
        ("MAE_return", "Return MAE", "MAE_by_horizon_V6.png"),
        ("Directional_Accuracy_pct", "Directional Accuracy (%)", "DA_by_horizon_V6.png"),
    ]:
        plt.figure(figsize=(11, 6))

        for model, g in metrics.groupby("model"):
            g = g.sort_values("horizon_days")
            plt.plot(
                g["horizon_days"],
                g[metric],
                marker="o",
                linewidth=1.5,
                label=model
            )

        plt.xscale("log")
        plt.xlabel("Forecast horizon (days, log scale)")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} across forecast horizons — V6")
        plt.legend(fontsize=7, ncol=2)
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / fname, dpi=200)
        plt.close()


# =============================================================================
# 26. SUMMARY
# =============================================================================

def write_summary(
    df: pd.DataFrame,
    metrics: pd.DataFrame,
    sign_df: pd.DataFrame,
    mcs_df: pd.DataFrame,
    comp_df: pd.DataFrame
) -> None:

    lines = []
    lines.append("=" * 90)
    lines.append("BITCOIN FORECASTING V6 — FULL DATA SUMMARY")
    lines.append("=" * 90)
    lines.append(f"Input: {INPUT_FILE}")
    lines.append(f"Rows: {len(df)}")
    lines.append(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    lines.append(f"Initial training: {INITIAL_TRAIN}")
    lines.append(f"Retraining interval: {RETRAIN_EVERY}")
    lines.append("Forecast mode: direct multi-horizon expanding walk-forward")
    lines.append("Forecast origins: ALL valid origins")
    lines.append("")

    lines.append("BEST MODEL BY RMSE")
    best = (
        metrics.sort_values("RMSE_return")
        .groupby("horizon", as_index=False)
        .first()
    )
    for _, r in best.iterrows():
        lines.append(
            f"{r['horizon']:>5}: {r['model']:<24} "
            f"RMSE={r['RMSE_return']:.6f}, "
            f"MAE={r['MAE_return']:.6f}"
        )

    lines.append("")
    lines.append("LONG-HORIZON POSITIVE-RATE AUDIT")
    for _, r in sign_df.iterrows():
        lines.append(
            f"{r['horizon']:>5}: {r['positive_pct']:.2f}% positive "
            f"({int(r['positive_n'])}/{int(r['n_origins'])}), "
            f"CI95=[{r['positive_rate_CI95_low_pct']:.2f}, "
            f"{r['positive_rate_CI95_high_pct']:.2f}], "
            f"binomial p={r['exact_binomial_p_vs_0.5']:.3e}"
        )

    lines.append("")
    lines.append("IMPORTANT INTERPRETATION")
    lines.append(
        "A high or 100% positive realized-return rate is an unconditional "
        "sample property. It is NOT treated as proof of conditional forecastability."
    )
    lines.append(
        "Long-horizon overlapping observations reduce the number of effectively "
        "independent pieces of information; HAC and block-bootstrap diagnostics "
        "are therefore interpreted cautiously."
    )

    lines.append("")
    lines.append("MCS SENSITIVITY")
    if len(mcs_df):
        for hname in HORIZONS:
            for alpha in MCS_ALPHAS:
                z = mcs_df[
                    (mcs_df["horizon"] == hname) &
                    (mcs_df["alpha"] == alpha)
                ]
                if len(z):
                    members = z.iloc[0]["MCS_members"]
                    lines.append(
                        f"{hname:>5}, alpha={alpha:.2f}: {members}"
                    )

    lines.append("")
    lines.append("COMPUTATIONAL AUDIT")
    for _, r in comp_df.iterrows():
        lines.append(
            f"{r['horizon']:>5}: forecasts={int(r['forecast_origins'])}, "
            f"fits={int(r['fit_count'])}, "
            f"runtime={r['total_horizon_runtime_sec']/60:.2f} min"
        )

    (RESULT_DIR / "summary_V6_2.txt").write_text(
        "\n".join(lines),
        encoding="utf-8"
    )


# =============================================================================
# 27. METADATA
# =============================================================================

def write_metadata(df: pd.DataFrame) -> None:
    metadata = {
        "version": "V6.2",
        "project": str(PROJECT_DIR),
        "input_file": str(INPUT_FILE),
        "input_sha256": sha256_file(INPUT_FILE),
        "rows": int(len(df)),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "horizons": HORIZONS,
        "initial_train": INITIAL_TRAIN,
        "retrain_every": RETRAIN_EVERY,
        "max_forecasts": MAX_FORECASTS,
        "random_seed": RANDOM_SEED,
        "multi_seed_sensitivity": MULTI_SEEDS if RUN_MULTI_SEED_SENSITIVITY else [],
        "models": MODEL_NAMES,
        "feature_columns": MODEL_FEATURES,
        "vs_lstm_feature_columns": VS_LSTM_FEATURES,
        "ensemble_components": ["Zero Return", "Historical Mean", "Historical Median", "ARIMA", "OLS", "GLS", "LSTM", "VS-LSTM", "GLS-LSTM"],
        "ensemble_weight_rule": "normalized inverse MAE over previous 20 OOS errors",
        "ensemble_exclusions": {"Random Walk with Drift": "identical to Historical Mean", "Majority Direction": "direction-only benchmark", "Persistence Direction": "direction-only benchmark"},
        "target_definition": "100*log(P[t+h]/P[t])",
        "walk_forward": "expanding",
        "first_valid_origin_rule": "INITIAL_TRAIN + horizon - 1",
        "training_target_end_rule": "origin - horizon + 1; y[origin-horizon] is observable at origin",
        "mcs_alphas": MCS_ALPHAS,
        "mcs_bootstraps": MCS_BOOTSTRAPS,
        "robustness_bootstraps": ROBUSTNESS_BOOTSTRAPS,
        "robustness_block_lengths": BOOTSTRAP_BLOCK_LENGTHS,
        "lstm_lookback": LSTM_LOOKBACK,
        "lstm_epochs": LSTM_EPOCHS,
        "lstm_batch_size": LSTM_BATCH_SIZE,
        "lstm_early_stopping_patience": LSTM_PATIENCE,
        "arima_order": ARIMA_ORDER,
        "glsar_rho": GLSAR_RHO,
        "glsar_maxiter": GLSAR_MAXITER,
        "tensorflow_available": TF_AVAILABLE,
        "python_version": sys.version,
        "platform": platform.platform(),
    }

    if TF_AVAILABLE:
        metadata["tensorflow_version"] = tf.__version__
        try:
            metadata["gpu_devices"] = [
                str(x) for x in tf.config.list_physical_devices("GPU")
            ]
        except Exception:
            metadata["gpu_devices"] = []

    (RESULT_DIR / "metadata_V6_2.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


# =============================================================================
# 28. CHECKPOINTING
# =============================================================================

def checkpoint_path(hname: str) -> Path:
    return CHECKPOINT_DIR / f"forecast_level_{hname}.csv"


def save_checkpoint(hname: str, result_df: pd.DataFrame) -> None:
    result_df.to_csv(
        checkpoint_path(hname),
        index=False,
        encoding="utf-8-sig"
    )


# =============================================================================
# 29. MAIN
# =============================================================================

def main() -> None:
    overall_start=time.perf_counter()
    log("="*90)
    log("BITCOIN FORECASTING V6.2.1 — PUBLICATION-GRADE FULL DATA RUN")
    log("="*90)
    log(f"Input: {INPUT_FILE}")
    log(f"MAX_FORECASTS={MAX_FORECASTS} (None=ALL valid origins)")
    log(f"INITIAL_TRAIN={INITIAL_TRAIN}, RETRAIN_EVERY={RETRAIN_EVERY}")
    log(f"Seed={RANDOM_SEED}, TensorFlow available={TF_AVAILABLE}")
    log(f"MCS: B={MCS_BOOTSTRAPS}, block={MCS_BLOCK_LENGTH}; robustness B={ROBUSTNESS_BOOTSTRAPS}")

    df=load_dataset()
    dataset_sha256=sha256_file(INPUT_FILE)
    all_results=[]; all_comp=[]

    for hname,h in HORIZONS.items():
        cp=checkpoint_path(hname)
        meta_path=CHECKPOINT_DIR/f"checkpoint_meta_{hname}.json"
        expected_cfg=checkpoint_config(hname,h,dataset_sha256)
        reused=False
        complete_set=set()
        if cp.exists() and meta_path.exists():
            try:
                saved_cfg=json.loads(meta_path.read_text(encoding="utf-8"))
                if saved_cfg==expected_cfg:
                    old=pd.read_csv(cp)
                    expected_origins=valid_forecast_origins(len(df),h,MAX_FORECASTS)
                    expected_set=set(expected_origins.tolist())
                    valid_complete=True
                    if not {"origin_index","model","actual_return","predicted_return"}.issubset(old.columns):
                        valid_complete=False
                    else:
                        grouped=old.groupby("origin_index")
                        complete_set=set()
                        for oi,g in grouped:
                            models_here=set(g["model"].astype(str))
                            if (
                                len(g)==len(MODEL_NAMES)
                                and models_here==set(MODEL_NAMES)
                            ):
                                complete_set.add(int(oi))
                        valid_complete = (
                            complete_set == expected_set
                            and len(old) == len(expected_set)*len(MODEL_NAMES)
                        )
                    done=len(complete_set) if "complete_set" in locals() else 0
                    if valid_complete:
                        log(f"{hname}: complete compatible checkpoint found ({done} origins); reusing.")
                        all_results.append(old)
                        cpath=CHECKPOINT_DIR/f"computational_checkpoint_{hname}.csv"
                        if cpath.exists(): all_comp.append(pd.read_csv(cpath))
                        reused=True
                    else:
                        log(f"{hname}: checkpoint is not a strict complete origin/model grid; resume validator will inspect it.")
                else:
                    log(f"{hname}: checkpoint configuration mismatch; old checkpoint ignored.")
            except Exception as exc:
                log(f"{hname}: checkpoint read failed; fresh/resume path will be used: {exc}")
        if reused:
            continue

        result_h,comp_h=run_horizon(df,hname,h,dataset_sha256)
        save_checkpoint(hname,result_h)
        result_h.to_csv(RESULT_DIR/f"forecasting_results_{hname}_V6_2.csv",index=False,encoding="utf-8-sig")
        all_results.append(result_h); all_comp.append(comp_h)

    if not all_results:
        raise RuntimeError("No forecast results were generated.")
    results=pd.concat(all_results,ignore_index=True)
    comp=pd.concat(all_comp,ignore_index=True) if all_comp else pd.DataFrame()

    metrics=add_skill_scores(compute_metrics(results))
    metrics.to_csv(RESULT_DIR/"forecasting_metrics_V6_2_multihorizon.csv",index=False,encoding="utf-8-sig")

    best=(metrics.sort_values(["horizon_days","RMSE_return"]).groupby("horizon",as_index=False).first())
    best.to_csv(RESULT_DIR/"best_model_by_horizon_V6_2.csv",index=False,encoding="utf-8-sig")
    ranking=metrics.copy()
    ranking["rank_RMSE"]=ranking.groupby("horizon")["RMSE_return"].rank(method="min")
    ranking["rank_MAE"]=ranking.groupby("horizon")["MAE_return"].rank(method="min")
    ranking.to_csv(RESULT_DIR/"model_ranking_by_horizon_V6_2.csv",index=False,encoding="utf-8-sig")

    sign_df=sign_distribution_analysis(results)
    sign_df.to_csv(RESULT_DIR/"sign_distribution_tests_V6_2.csv",index=False,encoding="utf-8-sig")
    calendar_df=calendar_positive_rate_analysis(df,results)
    calendar_df.to_csv(RESULT_DIR/"calendar_positive_rate_V6_2.csv",index=False,encoding="utf-8-sig")

    dm_df=dm_pairwise(results)
    dm_df.to_csv(RESULT_DIR/"diebold_mariano_pairwise_HAC_V6_2.csv",index=False,encoding="utf-8-sig")
    best_dm=dm_best_nonensemble(results,metrics)
    best_dm.to_csv(RESULT_DIR/"dm_adaptive_vs_best_nonensemble_V6_2.csv",index=False,encoding="utf-8-sig")

    stat_df,stab_df=stationarity_break_diagnostics(df,results)
    stat_df.to_csv(RESULT_DIR/"stationarity_ADF_KPSS_V6_2.csv",index=False,encoding="utf-8-sig")
    stab_df.to_csv(RESULT_DIR/"stability_CUSUM_V6_2.csv",index=False,encoding="utf-8-sig")

    mcs_df=mcs_sensitivity(results)
    mcs_df.to_csv(RESULT_DIR/"model_confidence_set_sensitivity_V6_2.csv",index=False,encoding="utf-8-sig")
    robust_df=robustness_analysis(results)
    robust_df.to_csv(RESULT_DIR/"robustness_nonoverlap_recent_V6_2.csv",index=False,encoding="utf-8-sig")
    comp_audit=computational_audit(comp)
    comp_audit.to_csv(RESULT_DIR/"computational_audit_V6_2.csv",index=False,encoding="utf-8-sig")
    ablation=ensemble_ablation(metrics)
    ablation.to_csv(RESULT_DIR/"ablation_summary_V6_2.csv",index=False,encoding="utf-8-sig")

    if RUN_MULTI_SEED_SENSITIVITY and TF_AVAILABLE:
        ms=run_multi_seed_sensitivity(df)
        ms.to_csv(RESULT_DIR/"multi_seed_LSTM_sensitivity_V6_2.csv",index=False,encoding="utf-8-sig")
        if len(ms):
            ms.groupby(["horizon","seed"]).agg(
                n=("origin_index","count"),
                RMSE=("squared_error",lambda x: math.sqrt(float(np.mean(x)))),
                MAE=("absolute_error","mean"),
                mean_fit_sec=("fit_runtime_sec","mean")
            ).reset_index().to_csv(RESULT_DIR/"multi_seed_LSTM_sensitivity_summary_V6_2.csv",index=False,encoding="utf-8-sig")

    make_figures(metrics)
    write_metadata(df)
    write_summary(df,metrics,sign_df,mcs_df,comp_audit)
    results.to_csv(RESULT_DIR/"forecasting_results_V6_2_multihorizon.csv",index=False,encoding="utf-8-sig")

    elapsed=time.perf_counter()-overall_start
    log("="*90)
    log(f"V6.2 FULL RUN COMPLETE — total runtime {elapsed/3600:.2f} hours")
    log(f"Results directory: {RESULT_DIR}")
    log("="*90)


if __name__=="__main__":
    main()
