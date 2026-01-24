import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env_trading_rl import TradingEnvRL

# ============================
# CONFIG
# ============================
CSV_PATH   = "regime_classified/1h/eth_regime_1h_h3_t0010.csv"
MODEL_PATH = "models_rl/1h/ppo_eth_regime_1h_h3_t0010"
VEC_PATH   = "models_rl/1h/vecnorm_eth_regime_1h_h3_t0010.pkl"

LOOKBACK_HOURS = 72
WARMUP_BARS    = 150
TOTAL_BARS     = LOOKBACK_HOURS + WARMUP_BARS

# ============================
# LOAD ONLY REQUIRED DATA
# ============================
df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])
df = df.tail(TOTAL_BARS).reset_index(drop=True)

# ============================
# ENV
# ============================
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

# ============================
# BACKTEST
# ============================
while env0.step_idx < len(df) - 1:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step([int(action[0])])

    row = df.iloc[env0.step_idx]

    log.append({
        "timestamp": row["timestamp"],
        "close": row["close"],
        "position": env0.position
    })

    if done[0]:
        break

log_df = pd.DataFrame(log)

# ============================
# KEEP ONLY LAST 72 HOURS (TIME-BASED)
# ============================
last_ts  = log_df["timestamp"].max()
start_ts = last_ts - pd.Timedelta(hours=LOOKBACK_HOURS)

log_df = log_df[
    (log_df["timestamp"] >= start_ts) &
    (log_df["timestamp"] <= last_ts)
].reset_index(drop=True)

# ============================
# PLOT
# ============================
fig, ax_price = plt.subplots(figsize=(14, 6))
ax_pos = ax_price.twinx()

ax_price.plot(
    log_df["timestamp"],
    log_df["close"],
    color="black",
    linewidth=1.8,
    label="Close"
)

ax_pos.step(
    log_df["timestamp"],
    log_df["position"],
    where="post",
    color="red",
    alpha=0.6,
    linewidth=1.5,
    label="Position"
)

ax_pos.set_ylim(-0.05, 1.05)
ax_pos.set_yticks([0, 1])
ax_pos.set_ylabel("Position")

# 🔥 CLAVE: cortar EXACTO el eje X
ax_price.set_xlim(start_ts, last_ts)

ax_price.set_title("ETH 1H — últimas 72 horas reales (PPO)")
ax_price.grid(True)
fig.autofmt_xdate()

plt.show()
