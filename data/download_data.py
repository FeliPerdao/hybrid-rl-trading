import ccxt
import pandas as pd
from datetime import datetime, timedelta
import os

DATA_DIR = "data"
SYMBOL = "ETH/USDT"

def fetch_ohlcv(symbol, timeframe, since_days=730):
    exchange = ccxt.binance({"enableRateLimit": True})

    since = exchange.parse8601(
        (datetime.utcnow() - timedelta(days=since_days)).isoformat()
    )

    all_ohlcv = []
    limit = 1000

    while True:
        ohlcv = exchange.fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            since=since,
            limit=limit
        )

        if not ohlcv:
            break

        all_ohlcv.extend(ohlcv)
        since = ohlcv[-1][0] + 1

        if len(ohlcv) < limit:
            break

    df = pd.DataFrame(
        all_ohlcv,
        columns=["timestamp", "open", "high", "low", "close", "volume"]
    )

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def save_ohlcv(df, timeframe):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"eth_ohlcv_{timeframe}.csv")
    df.to_csv(path, index=False)
    print(f"✔ Guardado: {path} ({len(df)} velas)")


def update_or_download(timeframe):
    path = os.path.join(DATA_DIR, f"eth_ohlcv_{timeframe}.csv")

    if not os.path.exists(path):
        print(f"⬇ Descargando ETH {timeframe}...")
        df = fetch_ohlcv(SYMBOL, timeframe)
        save_ohlcv(df, timeframe)
        return

    print(f"🔄 Actualizando ETH {timeframe}...")
    df_old = pd.read_csv(path)
    df_old["timestamp"] = pd.to_datetime(df_old["timestamp"])

    last_ts = df_old["timestamp"].max()
    since_days = max(1, (datetime.utcnow() - last_ts).days + 1)

    df_new = fetch_ohlcv(SYMBOL, timeframe, since_days=since_days)

    df = pd.concat([df_old, df_new])
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")

    save_ohlcv(df, timeframe)


def run():
    for tf in ["30m", "1h"]:
        update_or_download(tf)
