import numpy as np
import pandas as pd
import joblib
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
REGIME_THRESHOLD = 0.7


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
    # ---- DATA ----
    df = pd.read_csv(DATA_PATH)

    feature_cols = joblib.load(FEATURES_REGIME)
    model_regime = joblib.load(MODEL_REGIME)

    features = df[feature_cols]

    # ---- REGIME (idéntico a training) ----
    regime_proba = model_regime.predict_proba(features)[:, 1]
    regime_on = (regime_proba > REGIME_THRESHOLD).astype(int)
    #regime_on[:] = 1

    # ---- ENV (MISMO input que TRAIN) ----
    prices = df["close"]

    env = TradingEnv(
        features_df=features,
        price_series=prices,
        regime_on=pd.Series(regime_on)
    )
    env.set_mode("backtest")
    
    obs = env.reset(start_idx=0)

    model = DQN.load(MODEL_PATH, env=env)

    done = False

    equity = [1.0]
    trades = []

    # ---- LOOP ----
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        pnl = info.get("pnl", 0.0)
        equity.append(equity[-1] + pnl)

        if pnl != 0:
            trades.append(pnl)

    # ---- CIERRE FORZADO ----
    if env.position == 1:
        last_price = df.iloc[min(env.step_idx, len(df) - 1)]["close"]
        pnl = (last_price - env.entry_price) / env.entry_price
        equity[-1] += pnl
        trades.append(pnl)

    equity = np.array(equity)
    trades = np.array(trades)

    # ---- REPORT ----
    print("\n====== BACKTEST ======")
    print(f"Final Equity: {equity[-1]:.4f}")
    print(f"Total PnL: {(equity[-1] - 1):.4f}")
    print(f"Trades: {len(trades)}")
    print("Trades PnL:", trades)


    if len(trades) > 0:
        print(f"Win rate: {(trades > 0).mean():.2%}")
        if (trades < 0).any():
            pf = trades[trades > 0].sum() / abs(trades[trades < 0].sum())
            print(f"Profit factor: {pf:.2f}")

    print(f"Max DD: {max_drawdown(equity):.4f}")

    # ---- LOG (solo métricas reales) ----
    log_backtest_result(
        final_equity=float(equity[-1]),
        total_pnl=float(equity[-1] - 1),
        trades=int(len(trades)),
        win_rate=float((trades > 0).mean()) if len(trades) > 0 else None,
        profit_factor=(
            float(trades[trades > 0].sum() / abs(trades[trades < 0].sum()))
            if (trades < 0).any()
            else None
        ),
        max_dd=float(max_drawdown(equity)),
        params={
            "REGIME_THRESHOLD": REGIME_THRESHOLD
        }
    )

    # ---- PLOT ----
    plt.figure(figsize=(10, 4))
    plt.plot(equity)
    plt.title("Equity Curve")
    plt.xlabel("Steps")
    plt.ylabel("Equity")
    plt.grid()
    plt.tight_layout()
    plt.show()


def run():
    main()
