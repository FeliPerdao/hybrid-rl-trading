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

    # ============================
    # RETURNS (momentum multi-scale)
    # ============================
    df["ret_1h"] = df["close"].pct_change()
    df["ret_4h"] = df["close"].pct_change(4)
    df["ret_12h"] = df["close"].pct_change(12)

    # ============================
    # CANDLE STRUCTURE
    # ============================
    df["range_pct"] = (df["high"] - df["low"]) / df["close"]
    df["body_pct"] = (df["close"] - df["open"]) / df["close"]

    # ============================
    # EMAS
    # ============================
    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    df["ema100"] = ema(df["close"], 100)

    # Slopes
    df["ema20_slope"] = df["ema20"].pct_change()

    # EMA structure (trend)
    df["ema_trend"] = (
        (df["ema20"] > df["ema50"]) &
        (df["ema50"] > df["ema100"])
    ).astype(int)

    # Distance to EMAs (THIS is what matters)
    df["dist_ema20"] = (df["close"] - df["ema20"]) / df["close"]
    df["dist_ema50"] = (df["close"] - df["ema50"]) / df["close"]
    df["dist_ema100"] = (df["close"] - df["ema100"]) / df["close"]

    # ============================
    # RSI
    # ============================
    df["rsi14"] = rsi(df["close"], 14)
    df["rsi_slope"] = df["rsi14"].diff()

    # ============================
    # ATR & VOLATILITY
    # ============================
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs()
    ], axis=1).max(axis=1)

    df["atr14"] = tr.rolling(14).mean()
    df["atr_pct"] = df["atr14"] / df["close"]

    # Volatility regime (z-score)
    atr_mean = df["atr_pct"].rolling(100).mean()
    atr_std  = df["atr_pct"].rolling(100).std()
    df["vol_z"] = (df["atr_pct"] - atr_mean) / atr_std

    # ============================
    # VOLUME (the missing piece)
    # ============================
    vol_ma = df["volume"].rolling(20).mean()
    df["vol_rel"] = df["volume"] / vol_ma

    # ============================
    df = df.dropna()
    return df


def run():
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df = build_features(df)
    df.to_csv(OUT_PATH, index=False)
    print(f"✔ Features guardadas: {OUT_PATH} ({len(df)} filas)")
