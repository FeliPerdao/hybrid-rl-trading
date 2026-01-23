import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env_trading_rl import TradingEnvRL
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

CSV_PATH = "regime_classified/1h/eth_regime_1h_h3_t0007.csv"
MODEL_PATH = "models/ppo_trading_v2"
VEC_PATH = "models/vec_normalize.pkl"

df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])

START_DATE = pd.Timestamp("2025-12-15 00:00:00")
END_DATE   = pd.Timestamp("2025-12-21 23:00:00")

start_idx = df.index[df["timestamp"] >= START_DATE][0]

def make_env():
    env = TradingEnvRL(df, eval_mode=True)
    env.step_idx = start_idx
    return env

env = DummyVecEnv([make_env])
env = VecNormalize.load(VEC_PATH, env)
env.training = False
env.norm_reward = False

env0 = env.envs[0].gym_env
model = PPO.load(MODEL_PATH)

obs = env.reset()

log = []

while True:
    row = df.iloc[env0.step_idx]

    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step([int(action[0])])

    log.append({
        "timestamp": row["timestamp"],
        "close": row["close"],
        "position": env0.position
    })

    if row["timestamp"] >= END_DATE or done[0]:
        break

log_df = pd.DataFrame(log)

log_df["date"] = log_df["timestamp"].dt.date
dates = sorted(log_df["date"].unique())

idx = 0

fig, ax_price = plt.subplots(figsize=(14, 6))
ax_pos = ax_price.twinx()

def plot_day(i):
    ax_price.clear()
    ax_pos.clear()

    day = dates[i]

    mask = (
        (log_df["timestamp"] >= pd.Timestamp(day) - pd.Timedelta(hours=72)) &
        (log_df["timestamp"] <= pd.Timestamp(day) + pd.Timedelta(hours=72))
    )

    sub = log_df[mask]

    ax_price.plot(sub["timestamp"], sub["close"], label="Close", color="black")
    ax_pos.step(sub["timestamp"], sub["position"], where="post", label="Position", color="red", alpha=0.6)

    ax_pos.set_ylim(-0.1, 1.1)
    ax_pos.set_yticks([0, 1])
    ax_pos.set_ylabel("Position")

    ax_price.set_title(f"ETH 1H – Contexto diario – {day}")
    ax_price.grid(True)

    fig.autofmt_xdate()

plot_day(idx)

ax_next = plt.axes([0.85, 0.02, 0.1, 0.05])
btn_next = Button(ax_next, "Siguiente día")

def next_day(event):
    global idx
    if idx < len(dates) - 1:
        idx += 1
        plot_day(idx)
        plt.draw()

btn_next.on_clicked(next_day)

plt.show()