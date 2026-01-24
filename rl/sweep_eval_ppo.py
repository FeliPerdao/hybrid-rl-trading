import os
import re
import time
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env_trading_rl import TradingEnvRL


MODELS_DIR = "models_rl"
DATA_DIR = "regime_classified"

AVAILABLE_TFS = ["1m","5m","15m","30m","1h","4h","1d"]
INITIAL_CAPITAL = 10_000
FEE = 0.0025


MODEL_RE = re.compile(
    r"ppo_eth_regime_(?P<tf>.+)_h(?P<h>\d+)_t(?P<t>\d+)\.zip"
)


def eval_model(tf, h, t):
    model_path = f"{MODELS_DIR}/{tf}/ppo_eth_regime_{tf}_h{h}_t{t}.zip"
    vec_path   = f"{MODELS_DIR}/{tf}/vecnorm_eth_regime_{tf}_h{h}_t{t}.pkl"
    data_path  = f"{DATA_DIR}/{tf}/eth_regime_{tf}_h{h}_t{t}.csv"

    if not all(map(os.path.exists, [model_path, vec_path, data_path])):
        print(f"  ⚠ faltan archivos para {tf} h{h} t{t}")
        return None

    df = pd.read_csv(data_path)

    def make_env():
        return TradingEnvRL(df, eval_mode=True)

    env = DummyVecEnv([make_env])
    env = VecNormalize.load(vec_path, env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(model_path)

    obs = env.reset()
    env0 = env.envs[0].gym_env

    capital = INITIAL_CAPITAL
    position = 0
    entry_price = None
    trades = 0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action[0])

        prev_position = env0.position
        obs, _, done, _ = env.step([action])
        new_position = env0.position

        price = df.iloc[env0.step_idx]["close"]

        # OPEN
        if prev_position == 0 and new_position == 1:
            entry_price = price

        # CLOSE
        if prev_position == 1 and new_position == 0:
            pnl_pct = (price / entry_price - 1)
            capital *= (1 + pnl_pct - 2 * FEE)
            trades += 1
            entry_price = None

        if done[0]:
            break

    ret_pct = (capital / INITIAL_CAPITAL - 1) * 100
    expectancy = (capital / INITIAL_CAPITAL) ** (1 / max(trades,1)) - 1

    return {
        "tf": tf,
        "h": h,
        "t": t,
        "capital_final": capital,
        "return_pct": ret_pct,
        "trades": trades,
        "expectancy_pct": expectancy * 100
    }


def main():
    t0 = time.time()
    results = []

    for tf in AVAILABLE_TFS:
        tf_dir = f"{MODELS_DIR}/{tf}"
        if not os.path.exists(tf_dir):
            continue

        print(f"\n===== TF {tf} =====")

        for fname in os.listdir(tf_dir):
            m = MODEL_RE.match(fname)
            if not m:
                continue

            h = m.group("h")
            t = m.group("t")

            print(f"→ Evaluando h{h} t{t}")
            res = eval_model(tf, h, t)
            if res:
                results.append(res)
                print(
                    f"  ✔ final={res['capital_final']:.2f} "
                    f"ret={res['return_pct']:.2f}% "
                    f"trades={res['trades']} "
                    f"exp={res['expectancy_pct']:.3f}%"
                )

    print("\n===== RESUMEN =====")
    for r in results:
        print(
            f"{r['tf']} h{r['h']} t{r['t']} | "
            f"{r['return_pct']:.2f}% | "
            f"trades={r['trades']} | "
            f"exp={r['expectancy_pct']:.3f}%"
        )

    print(f"\nTiempo total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
