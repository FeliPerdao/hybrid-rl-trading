import os
import re
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

MODELS_DIR = "ml/models"
FEATURES_DIR = "features"
OUTPUT_DIR = "ml/filter_backtest"

os.makedirs(OUTPUT_DIR, exist_ok=True)

DROP_COLS = [
    "timestamp",
    "open","high","low","close",
    "ema20","ema50","ema100",
    "atr14"
]

MODEL_REGEX = re.compile(r"h(\d+)_t(\d+)")

def parse_ht(name):
    m = MODEL_REGEX.search(name)
    if not m:
        return None, None
    h = int(m.group(1))
    t = int(m.group(2)) / 1000
    return h, t

for tf in os.listdir(MODELS_DIR):
    tf_dir = os.path.join(MODELS_DIR, tf)
    if not os.path.isdir(tf_dir):
        continue

    print(f"\n📊 Backtesting filtro ML – TF {tf}")

    df = pd.read_csv(f"{FEATURES_DIR}/eth_features_{tf}.csv")
    X = df.drop(columns=DROP_COLS, errors="ignore")

    rows = []

    for file in os.listdir(tf_dir):
        if not file.startswith("model_regime_"):
            continue

        model_path = os.path.join(tf_dir, file)
        model = joblib.load(model_path)

        h, t = parse_ht(file)
        if h is None:
            continue

        preds = model.predict(X)

        total = len(preds)
        c0 = np.sum(preds == 0)
        c1 = np.sum(preds == 1)
        c2 = np.sum(preds == 2)

        rows.append({
            "tf": tf,
            "horizon": h,
            "threshold": t,
            "samples": total,
            "short_count": c0,
            "flat_count": c1,
            "long_count": c2,
            "short_pct": c0 / total,
            "flat_pct": c1 / total,
            "long_pct": c2 / total
        })

    result = pd.DataFrame(rows)
    result = result.sort_values(["horizon","threshold"])

    # CSV por TF
    csv_path = f"{OUTPUT_DIR}/filter_summary_{tf}.csv"
    result.to_csv(csv_path, index=False)
    print(f"✔ CSV guardado: {csv_path}")

    # Heatmap LONG %
    pivot = result.pivot(
        index="horizon",
        columns="threshold",
        values="long_pct"
    )

    plt.figure(figsize=(10,6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis")
    plt.title(f"{tf} – LONG frequency")
    plt.tight_layout()

    img_path = f"{OUTPUT_DIR}/heatmap_long_{tf}.png"
    plt.savefig(img_path)
    plt.close()

    print(f"🖼 Heatmap guardado: {img_path}")

print("\n🏁 Backtest de filtro ML finalizado.")
