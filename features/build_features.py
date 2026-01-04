import pandas as pd
#import numpy as np
#import os

DATA_PATH = "data/eth_ohlcv_1h.csv"
OUT_PATH = "features/eth_features_1h.csv"


def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def build_features(df):
    df = df.copy()

    # Returns
    df["ret_1h"] = df["close"].pct_change()
    df["ret_4h"] = df["close"].pct_change(4)
    df["ret_12h"] = df["close"].pct_change(12)

    # Candle structure
    df["range_pct"] = (df["high"] - df["low"]) / df["close"]
    df["body_pct"] = (df["close"] - df["open"]) / df["close"]

    # EMAs
    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    df["ema100"] = ema(df["close"], 100)

    df["ema20_slope"] = df["ema20"].pct_change()
    df["ema_trend"] = (df["ema20"] > df["ema50"]).astype(int)

    # RSI
    df["rsi14"] = rsi(df["close"], 14)
    df["rsi_slope"] = df["rsi14"].diff()

    # ATR
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs()
    ], axis=1).max(axis=1)

    df["atr14"] = tr.rolling(14).mean()
    df["atr_pct"] = df["atr14"] / df["close"]

    df = df.dropna()
    return df


def run():
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df = build_features(df)
    df.to_csv(OUT_PATH, index=False)
    print(f"✔ Features guardadas: {OUT_PATH} ({len(df)} filas)")
