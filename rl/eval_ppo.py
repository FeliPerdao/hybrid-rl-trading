from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env_trading_rl import TradingEnvRL
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "regime_classified/1h/eth_regime_1h_h3_t0010.csv"
MODEL_PATH = "models_rl/1h/ppo_eth_regime_1h_h3_t0010"
VEC_PATH = "models_rl/1h/vecnorm_eth_regime_1h_h3_t0010.pkl"

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

FEE = 0.0025

# ================== CONTABILIDAD SEPARADA ==================
capital_virtual = env0.initial_balance

opens = 0
closes = 0

pnl_buckets = {
    "lt_-0.5": 0,
    "mid_-0.5_0.5": 0,
    "gt_0.5": 0,
}

trade_entry_price = None
trade_entry_capital = None

# ================== LOOP ==================
while True:
    action, _ = model.predict(obs, deterministic=True)
    action = int(action[0])
    actions_count[action] += 1

    prev_position = env0.position
    obs, reward, done, _ = env.step([action])
    new_position = env0.position
    
    balance_curve.append(capital_virtual)

    price = df.iloc[env0.step_idx]["close"]

    # ===== OPEN =====
    if prev_position == 0 and new_position == 1 and action == 1:
        opens += 1
        trade_entry_price = price
        trade_entry_capital = capital_virtual

        print(
            f"[OPEN ] #{opens:04d} step={env0.step_idx} "
            f"price={price:.2f} capital={capital_virtual:.2f}"
        )

    # ===== CLOSE =====
    if prev_position == 1 and new_position == 0 and action == 2:
        closes += 1

        pnl_pct = (price / trade_entry_price - 1) * 100
        capital_virtual *= (1 + pnl_pct/100 - 2*FEE)


        # buckets
        if pnl_pct < -0.5:
            pnl_buckets["lt_-0.5"] += 1
        elif pnl_pct > 0.5:
            pnl_buckets["gt_0.5"] += 1
        else:
            pnl_buckets["mid_-0.5_0.5"] += 1

        print(
            f"[CLOSE] #{closes:04d} step={env0.step_idx} "
            f"price={price:.2f} pnl={pnl_pct:+.2f}% "
            f"capital={capital_virtual:.2f}"
        )

        trade_entry_price = None
        trade_entry_capital = None

    if done[0]:
        break

# ================== RESULTADOS ==================
print("\n====== BACKTEST PPO (CONTABLE REAL) ======")
print(f"Capital inicial : {env0.initial_balance:,.2f}")
print(f"Capital final   : {capital_virtual:,.2f}")
print(f"Return total    : {(capital_virtual / env0.initial_balance - 1) * 100:.2f}%")
print()
print("Trades:")
print(f"  Opens : {opens}")
print(f"  Closes: {closes}")
print()
print("Distribución PnL:")
print(f"  < -0.5%        : {pnl_buckets['lt_-0.5']}")
print(f"  -0.5% a 0.5%   : {pnl_buckets['mid_-0.5_0.5']}")
print(f"  > 0.5%         : {pnl_buckets['gt_0.5']}")
print()
print("Acciones:", actions_count)

avg_pnl = (capital_virtual / env0.initial_balance) ** (1 / closes) - 1
print(f"Expectancy promedio: {avg_pnl*100:.3f}%")

plt.figure(figsize=(12, 4))
plt.plot(balance_curve)
plt.title("PPO Agent - Equity Curve")
plt.grid(True)
plt.show()
