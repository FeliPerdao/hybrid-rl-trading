import pandas as pd
import numpy as np
import joblib
import time
import os

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ==============================
# Available timeframes
# ==============================

AVAILABLE_TFS = ["1m","5m","15m","30m","1h","4h","1d"]

# ==============================
# Features
# ==============================

DROP_COLS = [
    "timestamp",
    "open","high","low","close",
    "ema20","ema50","ema100",
    "atr14"
]

# ==============================
# Target
# ==============================

def build_target(df, horizon, threshold):
    future_close = df["close"].shift(-horizon)
    future_ret = (future_close / df["close"]) - 1

    target = np.ones(len(df), dtype=int)
    target[future_ret < -threshold] = 0
    target[future_ret > threshold] = 2

    return pd.Series(target, index=df.index)

# ==============================
# CLI helpers
# ==============================

def ask_timeframes():
    selected = []
    print("\nSeleccioná timeframes a entrenar:\n")

    for tf in AVAILABLE_TFS:
        ans = input(f"¿Entrenar {tf}? [y/n] ").strip().lower()
        if ans == "y":
            selected.append(tf)

    if not selected:
        print("❌ No seleccionaste ningún timeframe. Chau.")
        exit(1)

    return selected

def ask_numbers(prompt, cast_fn):
    raw = input(prompt).strip()
    try:
        return [cast_fn(x) for x in raw.split(",")]
    except:
        print("❌ Entrada inválida. Usá números separados por coma.")
        exit(1)

def ask_use_raw():
    ans = input("\n¿Entrenar con features crudas (raw)? [y/n] ").strip().lower()
    return ans == "y"

# ==============================
# Train single config
# ==============================

def train_one(tf, horizon, threshold, features_dir):
    t0 = time.time()

    tag = f"{tf}_h{horizon}_t{int(threshold*1000):04d}"

    DATA_PATH = f"{features_dir}/eth_features_{tf}.csv"

    TF_MODELS_DIR = f"ml/models/{tf}"
    os.makedirs(TF_MODELS_DIR, exist_ok=True)

    MODEL_PATH = f"{TF_MODELS_DIR}/model_regime_{tag}.pkl"
    FEATURES_PATH = f"{TF_MODELS_DIR}/features_regime_{tag}.pkl"

    print(f"\n🚀 Training {tag}")
    print(f"📄 Dataset: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    df["target"] = build_target(df, horizon, threshold)
    df = df.dropna().reset_index(drop=True)

    features = df.drop(columns=DROP_COLS + ["target"])
    target = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.25,
        shuffle=False
    )

    classes = np.array([0,1,2])
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))

    model = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )

    model.fit(X_train, y_train, sample_weight=y_train.map(class_weights))

    preds = model.predict(X_test)

    print("====== REGIME FILTER AUDIT ======")
    print(classification_report(y_test, preds, digits=3))

    joblib.dump(model, MODEL_PATH)
    joblib.dump(features.columns.tolist(), FEATURES_PATH)

    elapsed = time.time() - t0
    print(f"✔ Modelo guardado: {MODEL_PATH}")
    print(f"⏱ Tiempo: {elapsed:.2f}s")

# ==============================
# Main
# ==============================

def main():
    start = time.time()

    tfs = ask_timeframes()
    horizons = ask_numbers("\nHorizons (ej: 5,6,7): ", int)
    thresholds = ask_numbers("\nThresholds (ej: 0.01,0.012): ", float)
    use_raw = ask_use_raw()

    features_dir = "features/raw" if use_raw else "features"

    print("\n🧠 TRAIN GRID CONFIG")
    print(f"Timeframes: {tfs}")
    print(f"Horizons: {horizons}")
    print(f"Thresholds: {thresholds}")
    print(f"Features dir: {features_dir}")

    for tf in tfs:
        for h in horizons:
            for t in thresholds:
                try:
                    train_one(tf, h, t, features_dir)
                except Exception as e:
                    print(f"💥 Error TF={tf} h={h} t={t}: {e}")

    total = time.time() - start
    print(f"\n🏁 Total grid training time: {total:.2f}s")

# ==============================
# CLI
# ==============================

if __name__ == "__main__":
    main()
