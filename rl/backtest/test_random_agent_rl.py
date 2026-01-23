import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from env_trading_rl import TradingEnvRL

# =========================
# Load classified data
# =========================

CSV_PATH = "regime_classified/1h/eth_regime_1h_h3_t0007.csv"

df = pd.read_csv(CSV_PATH)

assert "regime" in df.columns, "❌ Falta columna regime"

# =========================
# Run random agent
# =========================

env = TradingEnvRL(df)

obs = env.reset()
done = False

equity_curve = [env.equity]
reward_curve = []
actions_count = {0: 0, 1: 0, 2: 0}

while not done:
    action = env.action_space.sample()
    actions_count[action] += 1

    obs, reward, done, _ = env.step(action)

    equity_curve.append(env.equity)
    reward_curve.append(reward)

# =========================
# Results
# =========================

print("====== RANDOM RL AGENT ======")
print(f"Capital inicial : {env.initial_balance:,.2f}")
print(f"Capital final   : {env.equity:,.2f}")
print(f"Return total    : {(env.equity / env.initial_balance - 1) * 100:.2f}%")
print("Acciones:", actions_count)

# =========================
# Plot
# =========================

plt.figure(figsize=(12, 4))
plt.plot(equity_curve)
plt.title("Random RL Agent - Equity Curve")
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 3))
plt.plot(reward_curve)
plt.title("Random RL Agent - Reward Curve")
plt.grid(True)
plt.show()
