import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env_trading_rl import TradingEnvRL
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox

CSV_PATH = "regime_classified/1h/eth_regime_1h_h3_t0010.csv"
MODEL_PATH = "models_rl/1h/ppo_eth_regime_1h_h3_t0010"
VEC_PATH = "models_rl/1h/vecnorm_eth_regime_1h_h3_t0010.pkl"

df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])

START_DATE = pd.Timestamp("2026-01-20 00:00:00")
END_DATE   = pd.Timestamp("2026-01-24 14:00:00")

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

# ============================
# BACKTEST LOOP (FIX
# ============================
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step([int(action[0])])

    row = df.iloc[env0.step_idx]

    log.append({
        "timestamp": row["timestamp"],
        "close": row["close"],
        "position": env0.position,
        "action": int(action[0]),
        "reward": reward[0]
    })

    if row["timestamp"] >= END_DATE or done[0]:
        break

log_df = pd.DataFrame(log)
log_df["date"] = log_df["timestamp"].dt.date
dates = sorted(log_df["date"].unique())

idx = 0

# ============================
# PLOT
# ============================
fig, ax_price = plt.subplots(figsize=(14, 6))
ax_pos = ax_price.twinx()

def plot_day(i):
    ax_price.clear()
    ax_pos.clear()

    day = dates[i]

    
    last_ts = log_df["timestamp"].max()

    start_ts = pd.Timestamp(day) - pd.Timedelta(hours=72)
    end_ts = min(pd.Timestamp(day) + pd.Timedelta(hours=72), last_ts)

    mask = (
        (log_df["timestamp"] >= start_ts) &
        (log_df["timestamp"] <= end_ts)
    )


    sub = log_df[mask]

    ax_price.plot(sub["timestamp"], sub["close"], color="black", label="Close")
    ax_pos.step(
        sub["timestamp"],
        sub["position"],
        where="post",
        color="red",
        alpha=0.6,
        label="Position"
    )

    ax_pos.set_ylim(-0.1, 1.1)
    ax_pos.set_yticks([0, 1])
    ax_pos.set_ylabel("Position")

    ax_price.set_title(f"ETH 1H – {day}")
    ax_price.grid(True)

    fig.autofmt_xdate()

plot_day(idx)

# ============================
# CONTROLES
# ============================
ax_prev = plt.axes([0.68, 0.02, 0.1, 0.05])
ax_next = plt.axes([0.80, 0.02, 0.1, 0.05])
ax_date = plt.axes([0.05, 0.02, 0.25, 0.05])

btn_prev = Button(ax_prev, "◀ Día anterior")
btn_next = Button(ax_next, "Día siguiente ▶")
txt_date = TextBox(ax_date, "Ir a fecha", initial="YYYY-MM-DD")

def prev_day(event):
    global idx
    if idx > 0:
        idx -= 1
        plot_day(idx)
        plt.draw()

def next_day(event):
    global idx
    if idx < len(dates) - 1:
        idx += 1
        plot_day(idx)
        plt.draw()

def go_to_date(text):
    global idx
    try:
        target = pd.to_datetime(text).date()
        if target in dates:
            idx = dates.index(target)
        else:
            idx = int(np.argmin([abs((d - target).days) for d in dates]))
        plot_day(idx)
        plt.draw()
    except Exception:
        print("❌ Fecha inválida. Usá YYYY-MM-DD")

btn_prev.on_clicked(prev_day)
btn_next.on_clicked(next_day)
txt_date.on_submit(go_to_date)

plt.show()
