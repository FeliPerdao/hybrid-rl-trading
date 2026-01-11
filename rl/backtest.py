import numpy as np
import pandas as pd
import joblib
import config
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from rl.env_trading import TradingEnv
from utils.logger import log_backtest_result

# =====================
# PATHS
# =====================
MODEL_PATH = "rl/dqn_trader_regime"
DATA_PATH = "features/eth_features_1h.csv"

MODEL_REGIME = "ml/model_regime.pkl"
FEATURES_REGIME = "ml/feature_cols_regime.pkl"
#REGIME_THRESHOLD = config.REGIME_THRESHOLD
#MAX_HOLD = config.MAX_HOLD

# =====================
# METRICS
# =====================
def max_drawdown(equity):
    peak = equity[0]
    dd = 0.0
    for x in equity:
        peak = max(peak, x)
        dd = min(dd, x - peak)
    return dd


# =====================
# BACKTEST
# =====================
def main():
    print("DEBUG REGIME_THRESHOLD =", config.REGIME_THRESHOLD) #check parameters
    # ---- DATA ----
    df = pd.read_csv(DATA_PATH)

    feature_cols = joblib.load(FEATURES_REGIME)
    model_regime = joblib.load(MODEL_REGIME)

    features = df[feature_cols]

    # ---- REGIME (idéntico a training) ----
    regime_proba = model_regime.predict_proba(features)[:, 1]
    regime_on = (regime_proba > config.REGIME_THRESHOLD).astype(int)

    # ---- ENV (MISMO input que TRAIN) ----
    prices = df["close"]

    env = TradingEnv(
        features_df=features,
        price_series=prices,
        regime_on=pd.Series(regime_on),
        max_hold=config.MAX_HOLD
    )
    env.set_mode("backtest")
    
    obs = env.reset()

    model = DQN.load(MODEL_PATH, env=env)

    done = False

    equity = [1.0]
    trades = []
    hold_times = [] 

    # ---- LOOP ----
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        pnl = info.get("pnl", 0.0)
        equity.append(equity[-1] * (1 + pnl))

        if pnl != 0:
            trades.append(pnl)
            hold_times.append(info["hold_time"])


    # ---- CIERRE FORZADO ----
    if env.position == 1:
        last_price = df.iloc[min(env.step_idx, len(df) - 1)]["close"]
        pnl = (last_price - env.entry_price) / env.entry_price
        equity[-1] *= (1 + pnl)
        trades.append(pnl)

    equity = np.array(equity)
    trades = np.array(trades)

    # ---- REPORT ----
    print("\n====== BACKTEST ======")
    print(f"Final Equity: {equity[-1]:.4f}")
    print(f"Total PnL: {(equity[-1] - 1):.4f}")
    print(f"Trades: {len(trades)}")
    print("Trades PnL:", trades)
    print(f"Avg hold time: {np.mean(hold_times):.2f}")

    if len(trades) > 0:
        print(f"Win rate: {(trades > 0).mean():.2%}")
        if (trades < 0).any():
            pf = trades[trades > 0].sum() / abs(trades[trades < 0].sum())
            print(f"Profit factor: {pf:.2f}")

    print(f"Max DD: {max_drawdown(equity):.4f}")

    # ---- LOG (solo métricas reales) ----
    log_backtest_result(
        final_equity=equity[-1],
        total_pnl=equity[-1] - 1,
        trades=len(trades),
        win_rate=(trades > 0).mean() if len(trades) > 0 else None,
        profit_factor=(
            trades[trades > 0].sum() / abs(trades[trades < 0].sum())
            if (trades < 0).any()
            else None
        ),
        max_dd=max_drawdown(equity),
        params={
            "REGIME_THRESHOLD": config.REGIME_THRESHOLD,
            "MAX_HOLD": config.MAX_HOLD,
            "PENALTY_FACTOR": config.PENALTY_FACTOR,
            "ML_TARGET": config.ML_TARGET,
            "ML_HORIZON": config.ML_HORIZON,
            "ML_TARGET_THRESHOLD": config.ML_TARGET_THRESHOLD,
            "TIMEFRAME": config.TIMEFRAME,
            "ASSET": config.ASSET
        }
    )

    # ---- PLOT ----
    # plt.figure(figsize=(10, 4))
    # plt.plot(equity)
    # plt.title("Equity Curve")
    # plt.xlabel("Steps")
    # plt.ylabel("Equity")
    # plt.grid()
    # plt.tight_layout()
    # plt.show()


def run():
    main()
