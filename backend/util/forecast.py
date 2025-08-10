from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional
import os
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

# Optional imports
try:
  from prophet import Prophet  # type: ignore
  _HAS_PROPHET = True
except Exception:
  _HAS_PROPHET = False

try:
  import tensorflow as tf  # type: ignore
  _HAS_TF = True
except Exception:
  _HAS_TF = False


def _ensure_cmdstan_installed() -> None:
  """Ensure cmdstan is installed for Prophet (cmdstanpy backend).

  This may take a few minutes on first run and requires a working C++ toolchain.
  """
  try:
    import cmdstanpy  # type: ignore
    try:
      path = cmdstanpy.cmdstan_path()  # type: ignore[attr-defined]
    except Exception:
      path = None
    if not path or not os.path.isdir(path):
      # Install cmdstan quietly; may download and compile
      cmdstanpy.install_cmdstan(quiet=True, progress=True)  # type: ignore[attr-defined]
  except Exception:
    # Best-effort; let caller handle failure
    pass


@dataclass
class ForecastResult:
  ticker: str
  model: str
  mae: float
  horizon_days: int
  last_price: float
  predictions: pd.DataFrame  # columns: ['date', 'pred']
  history: pd.DataFrame      # columns: ['date', 'close']


def load_prices(ticker: str, period: str = "5y") -> pd.DataFrame:
  df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
  df = df.reset_index()[["Date", "Close"]]
  df.columns = ["date", "close"]
  return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
  out = df.copy()
  out["ret_1d"] = out["close"].pct_change()
  out["ret_5d"] = out["close"].pct_change(5)
  out["ma_5"] = out["close"].rolling(5).mean()
  out["ma_20"] = out["close"].rolling(20).mean()
  out["vol_20"] = out["close"].rolling(20).std()
  out = out.dropna().reset_index(drop=True)
  return out


def train_rf_forecast(df_feat: pd.DataFrame, horizon: int = 5) -> Dict[str, Any]:
  df = df_feat.copy()
  df["target"] = df["close"].shift(-horizon)
  df = df.dropna().reset_index(drop=True)

  X = df[["ret_1d", "ret_5d", "ma_5", "ma_20", "vol_20"]].values
  y = df["target"].values

  tscv = TimeSeriesSplit(n_splits=5)
  maes = []
  for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    maes.append(mean_absolute_error(y_test, pred))

  final_model = RandomForestRegressor(n_estimators=500, random_state=42)
  final_model.fit(X, y)

  return {
    "model": final_model,
    "mae": float(np.mean(maes)) if maes else float("nan"),
  }


def forecast_next(df_feat: pd.DataFrame, model: RandomForestRegressor, horizon: int = 5) -> pd.DataFrame:
  row = df_feat[["ret_1d", "ret_5d", "ma_5", "ma_20", "vol_20"]].iloc[-1:].values
  preds = []
  last_date = df_feat["date"].iloc[-1]
  for i in range(1, horizon + 1):
    yhat = model.predict(row)[0]
    last_date = last_date + pd.Timedelta(days=1)
    preds.append({"date": last_date, "pred": float(yhat)})
  return pd.DataFrame(preds)


def prophet_forecast(prices: pd.DataFrame, horizon: int = 5) -> Dict[str, Any]:
  if not _HAS_PROPHET:
    raise RuntimeError("prophet not installed")
  df = prices.rename(columns={"date": "ds", "close": "y"})
  # Holdout for MAE computation
  holdout = min(max(horizon, 5), 30)
  train_df = df.iloc[:-holdout].copy() if len(df) > holdout else df.copy()

  # Model for MAE on holdout
  m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
  try:
    m.fit(train_df)
  except Exception:
    _ensure_cmdstan_installed()
    m.fit(train_df)

  if len(df) > holdout:
    hold_future = m.predict(df[["ds"]].iloc[-holdout:])
    mae = float(mean_absolute_error(df["y"].iloc[-holdout:], hold_future["yhat"].iloc[-holdout:]))
  else:
    mae = float("nan")

  # Separate model trained on full data to forecast strictly after the last historical date
  m_full = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
  try:
    m_full.fit(df)
  except Exception:
    _ensure_cmdstan_installed()
    m_full.fit(df)

  future_full = m_full.make_future_dataframe(periods=horizon)
  fcst_full = m_full.predict(future_full)
  tail = fcst_full.tail(horizon)
  preds = tail[["ds", "yhat"]].rename(columns={"ds": "date", "yhat": "pred"}).copy()

  return {"preds": preds, "mae": mae}


def lstm_forecast(prices: pd.DataFrame, horizon: int = 5, window: int = 20) -> Dict[str, Any]:
  if not _HAS_TF:
    raise RuntimeError("tensorflow not installed")
  # Prepare data
  data = prices["close"].values.reshape(-1, 1).astype("float32")
  # Min-max scale
  min_v, max_v = float(np.min(data)), float(np.max(data))
  rng = max_v - min_v if max_v > min_v else 1.0
  data_s = (data - min_v) / rng

  X, y = [], []
  for i in range(window, len(data_s)):
    X.append(data_s[i - window:i])
    y.append(data_s[i])
  X, y = np.array(X), np.array(y)

  if len(X) < 30:
    raise RuntimeError("not enough data for LSTM")

  # Train/val split (last 20% as val)
  split = int(len(X) * 0.8)
  X_train, X_val = X[:split], X[split:]
  y_train, y_val = y[:split], y[split:]

  model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(window, 1)),
    tf.keras.layers.LSTM(32, return_sequences=False),
    tf.keras.layers.Dense(1)
  ])
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss="mae")
  model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=6, batch_size=32, verbose=0)

  # Validation MAE
  val_pred = model.predict(X_val, verbose=0)
  mae = float(np.mean(np.abs(val_pred - y_val)))

  # Forecast next horizon iteratively, anchored to last observed close to avoid jumps
  last_seq = data_s[-window:].copy()
  preds = []
  last_date = prices["date"].iloc[-1]
  last_close = float(prices["close"].iloc[-1])
  anchor_offset_abs: Optional[float] = None
  anchor_offset_s: Optional[float] = None
  for _ in range(horizon):
    yhat_s = float(model.predict(last_seq.reshape(1, window, 1), verbose=0)[0, 0])
    yhat = yhat_s * rng + min_v

    # On first step, compute offset so first prediction matches last_close
    if anchor_offset_abs is None:
      anchor_offset_abs = last_close - yhat
      anchor_offset_s = anchor_offset_abs / rng if rng != 0 else 0.0

    # Apply offset to keep continuity
    yhat_s_adj = yhat_s + (anchor_offset_s or 0.0)
    yhat_adj = yhat + (anchor_offset_abs or 0.0)

    last_date = last_date + pd.Timedelta(days=1)
    preds.append({"date": last_date, "pred": float(yhat_adj)})

    # Feed adjusted normalized value back into sequence
    last_seq = np.concatenate([last_seq[1:], np.array([[yhat_s_adj]], dtype="float32")], axis=0)

  return {"preds": pd.DataFrame(preds), "mae": mae}


def run_forecast(ticker: str, period: str = "5y", horizon: int = 5, model: str = "rf") -> ForecastResult:
  prices = load_prices(ticker, period)
  model_name: str = ""
  mae: float = float("nan")
  preds_df: Optional[pd.DataFrame] = None

  try:
    if model == "prophet":
      out = prophet_forecast(prices, horizon=horizon)
      preds_df = out["preds"]
      mae = float(out["mae"])
      model_name = "Prophet"
    elif model == "lstm":
      out = lstm_forecast(prices, horizon=horizon)
      preds_df = out["preds"]
      mae = float(out["mae"])
      model_name = "LSTM"
    else:
      # Random Forest default
      feat = build_features(prices)
      trained = train_rf_forecast(feat, horizon=horizon)
      preds_df = forecast_next(feat, trained["model"], horizon=horizon)
      mae = float(trained["mae"])
      model_name = "RandomForestRegressor"
  except Exception:
    # Fallback to RF if alt model fails
    feat = build_features(prices)
    trained = train_rf_forecast(feat, horizon=horizon)
    preds_df = forecast_next(feat, trained["model"], horizon=horizon)
    mae = float(trained["mae"])
    if not model_name:
      model_name = "RandomForestRegressor (fallback)"

  return ForecastResult(
    ticker=ticker,
    model=model_name,
    mae=mae,
    horizon_days=horizon,
    last_price=float(prices["close"].iloc[-1]),
    predictions=preds_df if preds_df is not None else pd.DataFrame(columns=["date", "pred"]),
    history=prices.tail(252*2).copy(),  # last ~2 years for context
  )


