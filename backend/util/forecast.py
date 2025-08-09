from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor


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


def run_forecast(ticker: str, period: str = "5y", horizon: int = 5) -> ForecastResult:
    prices = load_prices(ticker, period)
    feat = build_features(prices)
    trained = train_rf_forecast(feat, horizon=horizon)
    pred_df = forecast_next(feat, trained["model"], horizon=horizon)
    return ForecastResult(
        ticker=ticker,
        model="RandomForestRegressor",
        mae=float(trained["mae"]),
        horizon_days=horizon,
        last_price=float(prices["close"].iloc[-1]),
        predictions=pred_df,
        history=prices.tail(252*2).copy(),  # last ~2 years for context
    )


