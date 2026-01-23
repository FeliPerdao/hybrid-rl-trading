import pandas as pd
import matplotlib.pyplot as plt
from env_trading_rl import TradingEnvRL

# =========================
# Load data (con regime)
# =========================

CSV_PATH = "regime_classified/1h/eth_regime_1h_h3_t0007.csv"

df = pd.read_csv(CSV_PATH)
assert "regime" in df.columns, "❌ Falta regime"

# =========================
# Rule-based agent
# =========================

env = TradingEnvRL(df)

obs = env.reset()
done = False

equity_curve = [env.equity]
actions_count = {0: 0, 1: 0, 2: 0}

while not done:
    regime = int(obs[-1])      # último feature
    position = int(obs[3])     # posición actual

    if regime == 2 and position == 0:
        action = 1  # OPEN
    elif regime == 0 and position == 1:
        action = 2  # CLOSE
    else:
        action = 0  # HOLD

    actions_count[action] += 1

    obs, reward, done, _ = env.step(action)
    equity_curve.append(env.equity)

# =========================
# Results
# =========================

print("====== RULE-BASED AGENT ======")
print(f"Capital inicial : {env.initial_balance:,.2f}")
print(f"Capital final   : {env.equity:,.2f}")
print(f"Return total    : {(env.equity / env.initial_balance - 1) * 100:.2f}%")
print("Acciones:", actions_count)

# =========================
# Plot
# =========================

plt.figure(figsize=(12, 4))
plt.plot(equity_curve)
plt.title("Rule-Based Agent (ML Regime Filter)")
plt.grid(True)
plt.show()
