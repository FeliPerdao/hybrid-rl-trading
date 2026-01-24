import os
import time
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env_trading_rl import TradingEnvRL

# ==============================
# CONFIG
# ==============================

AVAILABLE_TFS = ["1m","5m","15m","30m","1h","4h","1d"]
BASE_DATA_DIR = "regime_classified"
MODELS_DIR = "models_rl"

TOTAL_TIMESTEPS = 1_000_000

# ==============================
# ENV FACTORY
# ==============================

def make_env(df):
    def _init():
        return TradingEnvRL(df)
    return _init

# ==============================
# CLI HELPERS
# ==============================

def ask_yes_no(msg):
    return input(msg).strip().lower() == "y"

def list_regime_files(tf):
    tf_dir = os.path.join(BASE_DATA_DIR, tf)
    if not os.path.isdir(tf_dir):
        return []

    return sorted([
        f for f in os.listdir(tf_dir)
        if f.endswith(".csv")
    ])

def select_regime_files(tf):
    files = list_regime_files(tf)
    if not files:
        print(f"⚠ No hay archivos en {BASE_DATA_DIR}/{tf}")
        return []

    selected = []

    while True:
        print(f"\n📂 Regimes disponibles para {tf}:")
        for i, f in enumerate(files):
            print(f"  [{i}] {f}")

        print("  [0] Terminar selección para este TF")

        try:
            idx = int(input("Elegí un archivo: "))
        except:
            print("❌ Número inválido")
            continue

        if idx == 0:
            break

        if idx < 0 or idx >= len(files):
            print("❌ Índice fuera de rango")
            continue

        selected.append(files[idx])
        print(f"✔ Agregado: {files[idx]}")

    return selected

# ==============================
# TRAIN ONE MODEL
# ==============================

def train_rl(tf, csv_file):
    tag = csv_file.replace(".csv", "")
    data_path = os.path.join(BASE_DATA_DIR, tf, csv_file)

    print(f"\n🚀 Entrenando RL → {tf} | {tag}")
    print(f"📄 Dataset: {data_path}")

    df = pd.read_csv(data_path)

    env = DummyVecEnv([make_env(df)])
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=lambda f: 3e-4 * f,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        clip_range=0.2,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1
    )

    t0 = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    elapsed = time.time() - t0

    tf_model_dir = os.path.join(MODELS_DIR, tf)
    os.makedirs(tf_model_dir, exist_ok=True)

    model_path = os.path.join(tf_model_dir, f"ppo_{tag}")
    vec_path = os.path.join(tf_model_dir, f"vecnorm_{tag}.pkl")

    model.save(model_path)
    env.save(vec_path)

    print(f"✔ Modelo guardado: {model_path}")
    print(f"⏱ Tiempo: {elapsed:.2f}s")

    return elapsed

# ==============================
# MAIN
# ==============================

def main():
    print("\n🧠 RL MULTI-MODEL TRAINER\n")

    selections = []  # (tf, csv)
    timings = []

    for tf in AVAILABLE_TFS:
        if not ask_yes_no(f"\n¿Entrenar TF {tf}? [y/n] "):
            continue

        selected_files = select_regime_files(tf)
        for f in selected_files:
            selections.append((tf, f))

    if not selections:
        print("\n❌ No seleccionaste nada. Abortando.")
        return

    print("\n📋 SELECCIÓN FINAL:")
    for tf, f in selections:
        print(f"  - {tf} → {f}")

    if not ask_yes_no("\n¿Confirmar y entrenar? [y/n] "):
        print("❌ Cancelado.")
        return

    print("\n🔥 INICIANDO ENTRENAMIENTOS\n")
    total_start = time.time()

    for tf, f in selections:
        elapsed = train_rl(tf, f)
        timings.append((tf, f, elapsed))

    total_elapsed = time.time() - total_start

    print("\n🏁 RESUMEN FINAL")
    for tf, f, t in timings:
        print(f"  {tf} | {f} → {t:.2f}s")

    print(f"\n⏱ Tiempo total: {total_elapsed:.2f}s")

# ==============================
# ENTRY
# ==============================

if __name__ == "__main__":
    main()
