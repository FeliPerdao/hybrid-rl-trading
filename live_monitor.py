import subprocess
import time
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import sys
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from rl.env_trading_rl import TradingEnvRL

# ============================
# CONFIG
# ============================

CSV_PATH   = "regime_classified/1h/eth_regime_1h_h3_t0010.csv"
MODEL_PATH = "models_rl/1h/ppo_eth_regime_1h_h3_t0010"
VEC_PATH   = "models_rl/1h/vecnorm_eth_regime_1h_h3_t0010.pkl"

LOOKBACK_HOURS = 72
WARMUP_BARS    = 150
TOTAL_BARS     = LOOKBACK_HOURS + WARMUP_BARS

PIPELINE_SCRIPTS = [
    ["python", "data/update_data.py"],
    ["python", "features/update_features.py"],
    ["python", "4_ml_clasification_regime.py"],
]

UPDATE_MINUTES = {1, 16, 31, 46}

# ============================
# SOUND
# ============================

def play_sound(kind):
    try:
        if sys.platform.startswith("win"):
            import winsound
            freq = 900 if kind == "OPEN" else 500
            winsound.Beep(freq, 300)
        else:
            print("\a", end="", flush=True)
    except:
        pass

# ============================
# TK WINDOW
# ============================

root = tk.Tk()
root.title("ETH PPO LIVE")
root.geometry("1400x800")

# ============================
# MATPLOTLIB INIT
# ============================

fig, ax_price = plt.subplots(figsize=(14, 7))
ax_pos = ax_price.twinx()

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

status_text = ax_price.text(
    0.5, 0.5, "",
    transform=ax_price.transAxes,
    ha="center", va="center",
    fontsize=20,
    bbox=dict(facecolor="white", alpha=0.9),
    visible=False
)

info_box = ax_price.text(
    0.01, 0.99, "",
    transform=ax_price.transAxes,
    va="top",
    ha="left",
    bbox=dict(facecolor="white", alpha=0.85)
)

# ============================
# GLOBAL STATE
# ============================

last_position_global = None

# ============================
# PIPELINE
# ============================

def run_pipeline():
    for cmd in PIPELINE_SCRIPTS:
        subprocess.run(cmd, check=True)

# ============================
# BACKTEST + PLOT
# ============================

def update_plot():
    global last_position_global

    ax_price.clear()
    ax_pos.clear()

    df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])
    df = df.tail(TOTAL_BARS).reset_index(drop=True)

    def make_env():
        env = TradingEnvRL(df, eval_mode=True)
        env.step_idx = WARMUP_BARS
        return env

    env = DummyVecEnv([make_env])
    env = VecNormalize.load(VEC_PATH, env)
    env.training = False
    env.norm_reward = False

    env0 = env.envs[0].gym_env
    model = PPO.load(MODEL_PATH)

    obs = env.reset()
    log = []

    # ---- TRACKING REAL ----
    last_open_price = None
    last_open_time  = None
    last_close_pct  = None

    while env0.step_idx < len(df) - 1:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step([int(action[0])])

        row = df.iloc[env0.step_idx]

        # OPEN
        if env0.position == 1 and last_open_price is None:
            last_open_price = row["close"]
            last_open_time  = row["timestamp"]

        # CLOSE
        if env0.position == 0 and last_open_price is not None:
            last_close_pct = (row["close"] - last_open_price) / last_open_price * 100
            last_open_price = None
            last_open_time  = None

        log.append({
            "timestamp": row["timestamp"],
            "close": row["close"],
            "position": env0.position
        })

        if done[0]:
            break

    log_df = pd.DataFrame(log)

    last_ts  = log_df["timestamp"].max()
    start_ts = last_ts - pd.Timedelta(hours=LOOKBACK_HOURS)
    log_df = log_df[log_df["timestamp"] >= start_ts]

    # ====================
    # PLOT
    # ====================

    ax_price.plot(log_df["timestamp"], log_df["close"], color="black", linewidth=1.8)
    ax_pos.step(log_df["timestamp"], log_df["position"], where="post", color="red", alpha=0.6)

    ax_price.set_xlim(start_ts, last_ts)
    ax_pos.set_ylim(-0.05, 1.05)
    ax_pos.set_yticks([0, 1])
    ax_price.grid(True)
    ax_price.set_title("ETH 1H — LIVE PPO (últimas 72h)")

    # ====================
    # INFO BOX
    # ====================

    last_row = log_df.iloc[-1]
    price_now = last_row["close"]

    action = "OPEN" if last_row["position"] == 1 else "CLOSE"

    info = [
        f"LAST ACTION : {action}",
        f"TIME        : {last_row['timestamp']}",
        f"PRICE       : {price_now:.2f}",
    ]

    if action == "OPEN" and last_open_price is not None:
        dist_pct = (price_now - last_open_price) / last_open_price * 100
        info.append(f"DIST OPEN % : {dist_pct:+.2f}%")
        info.append(f"OPEN PRICE : {last_open_price:.2f}")

    if action == "CLOSE" and last_close_pct is not None:
        info.append(f"LAST TRADE %: {last_close_pct:+.2f}%")

    info_box.set_text("\n".join(info))

    # ====================
    # SOUND
    # ====================

    if last_position_global is not None and last_row["position"] != last_position_global:
        play_sound(action)

    last_position_global = last_row["position"]

    canvas.draw_idle()

# ============================
# MAIN LOOP
# ============================

last_run_minute = None

def scheduler():
    global last_run_minute

    now = datetime.now()

    if now.minute in UPDATE_MINUTES and now.minute != last_run_minute:
        last_run_minute = now.minute
        try:
            run_pipeline()
            update_plot()
        except Exception as e:
            print("ERROR:", e)

    root.after(1000, scheduler)  # chequea cada 1s

# ============================
# START APP
# ============================

status_text.set_text("Inicializando...")
status_text.set_visible(True)
canvas.draw_idle()

run_pipeline()
update_plot()

status_text.set_visible(False)
canvas.draw_idle()

scheduler()
root.mainloop()
