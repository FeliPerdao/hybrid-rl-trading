import pandas as pd
import numpy as np
import joblib

# ==============================
# Backtest ML (NO training)
# ==============================

def backtest_one(timeframe, horizon, threshold):
    # ⚠️ horizon y threshold son SOLO METADATA
    # el modelo YA FUE entrenado con esos valores

    # 1️⃣ Cargar datos
    df = pd.read_csv(f"features/eth_features_{timeframe}.csv")
    df = df.dropna().reset_index(drop=True)

    # 2️⃣ Cargar modelo EXISTENTE
    model = joblib.load(f"ml/model_regime_{timeframe}.pkl")
    features = joblib.load(f"ml/feature_cols_regime_{timeframe}.pkl")

    X = df[features]
    preds = model.predict(X)

    # 3️⃣ Retorno futuro real (solo para PnL)
    future_close = df["close"].shift(-horizon)
    future_ret = (future_close / df["close"]) - 1

    # 4️⃣ Estrategia
    strat_ret = np.zeros(len(df))

    strat_ret[preds == 2] = future_ret[preds == 2]     # long
    strat_ret[preds == 0] = -future_ret[preds == 0]    # short
    strat_ret[preds == 1] = 0.0                         # flat

    strat_ret = pd.Series(strat_ret).fillna(0)

    # 5️⃣ Métricas
    total_return = (1 + strat_ret).prod() - 1
    avg_ret = strat_ret.mean()
    winrate = (strat_ret > 0).mean()
    trades = (preds != 1).sum()

    return {
        "timeframe": timeframe,
        "model_horizon": horizon,
        "model_threshold": threshold,

        "total_return": total_return,
        "avg_return": avg_ret,
        "winrate": winrate,
        "trades": int(trades),
    }
