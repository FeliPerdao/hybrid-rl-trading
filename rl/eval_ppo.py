from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env_trading_rl import TradingEnvRL
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "regime_classified/1h/eth_regime_1h_h3_t0007.csv"
MODEL_PATH = "models/ppo_trading_v2"
VEC_PATH = "models/vec_normalize.pkl"

df = pd.read_csv(CSV_PATH)

def make_env():
    return TradingEnvRL(df, eval_mode=True)

env = DummyVecEnv([make_env])
env = VecNormalize.load(VEC_PATH, env)
env.training = False
env.norm_reward = False

env0 = env.envs[0].gym_env
model = PPO.load(MODEL_PATH)

obs = env.reset()

balance_curve = []
actions_count = {0: 0, 1: 0, 2: 0}

prev_position = env0.position
trade_id = 0
trade_entry_value = None
trade_entry_price = None

while True:
    action, _ = model.predict(obs, deterministic=True)
    action = int(action[0])
    actions_count[action] += 1

    prev_balance = env0.balance

    obs, reward, done, _ = env.step([action])

    new_position = env0.position
    price = df.iloc[env0.step_idx]["close"]

    # equity real (spot full allocation)
    equity = env0.balance + env0.position_value
    balance_curve.append(equity)

    # ========= OPEN =========
    if prev_position == 0 and new_position == 1 and action == 1:
        trade_id += 1
        trade_entry_value = prev_balance
        trade_entry_price = price

        print(
            f"[OPEN ] #{trade_id:03d} step={env0.step_idx} "
            f"price={price:.2f} value={trade_entry_value:.2f}"
        )

    # ========= CLOSE =========
    if prev_position == 1 and new_position == 0 and action == 2:
        trade_exit_value = env0.balance
        pnl_pct = (trade_exit_value / trade_entry_value - 1) * 100

        print(
            f"[CLOSE] #{trade_id:03d} step={env0.step_idx} "
            f"price={price:.2f} pnl={pnl_pct:+.2f}% value={trade_exit_value:.2f}"
        )

        trade_entry_value = None
        trade_entry_price = None

    prev_position = new_position

    if done[0]:
        final_equity = equity
        break

print("\n====== PPO AGENT ======")
print(f"Capital inicial : {env0.initial_balance:,.2f}")
print(f"Capital final   : {final_equity:,.2f}")
print(f"Return total    : {(final_equity / env0.initial_balance - 1) * 100:.2f}%")
print("Acciones:", actions_count)

plt.figure(figsize=(12, 4))
plt.plot(balance_curve)
plt.title("PPO Agent - Equity Curve")
plt.grid(True)
plt.show()
