import os
import re
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report

# =========================
# CONFIG
# =========================
MODELS_ROOT = "ml/models"
FEATURES_DIR = "features"
OUTPUT_DIR = "ml/audit"

os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_RE = re.compile(r"model_regime_(\w+)_h(\d+)_t(\d+)\.pkl")

DROP_COLS = [
    "timestamp",
    "open","high","low","close",
    "ema20","ema50","ema100",
    "atr14"
]

# =========================
# Target reconstruction
# =========================
def build_target(df, horizon, threshold):
    future_close = df["close"].shift(-horizon)
    future_ret = (future_close / df["close"]) - 1

    target = np.ones(len(df), dtype=int)
    target[future_ret < -threshold] = 0
    target[future_ret > threshold] = 2

    return target

# =========================
# MAIN AUDIT
# =========================
all_results = []

for tf in os.listdir(MODELS_ROOT):
    tf_dir = os.path.join(MODELS_ROOT, tf)
    if not os.path.isdir(tf_dir):
        continue

    print(f"\n🔍 Auditando TF={tf}")

    data_path = f"{FEATURES_DIR}/eth_features_{tf}.csv"
    df = pd.read_csv(data_path)

    tf_results = []

    for fname in os.listdir(tf_dir):
        m = MODEL_RE.match(fname)
        if not m:
            continue

        _, h, t = m.groups()
        h = int(h)
        t = int(t) / 1000

        model_path = os.path.join(tf_dir, fname)
        feats_path = model_path.replace("model_", "features_")

        model = joblib.load(model_path)
        cols = joblib.load(feats_path)

        df_work = df.copy()
        df_work["target"] = build_target(df_work, h, t)
        df_work = df_work.dropna().reset_index(drop=True)

        X = df_work[cols]
        y = df_work["target"]

        preds = model.predict(X)
        rep = classification_report(y, preds, output_dict=True)

        row = {
            "tf": tf,
            "horizon": h,
            "threshold": t,
            "accuracy": rep["accuracy"],
            "f1_macro": rep["macro avg"]["f1-score"],
            "f1_weighted": rep["weighted avg"]["f1-score"],
            "precision_0": rep["0"]["precision"],
            "recall_0": rep["0"]["recall"],
            "precision_2": rep["2"]["precision"],
            "recall_2": rep["2"]["recall"],
        }

        tf_results.append(row)
        all_results.append(row)

    # =========================
    # Export CSV per TF
    # =========================
    df_tf = pd.DataFrame(tf_results)
    tf_csv = f"{OUTPUT_DIR}/audit_{tf}.csv"
    df_tf.to_csv(tf_csv, index=False)
    print(f"📄 CSV TF guardado: {tf_csv}")

    # =========================
    # Heatmap H × T
    # =========================
    pivot = df_tf.pivot(
        index="horizon",
        columns="threshold",
        values="f1_macro"
    )

    plt.figure(figsize=(10,6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis")
    plt.title(f"TF {tf} – F1 Macro (H × T)")
    plt.tight_layout()

    heatmap_path = f"{OUTPUT_DIR}/heatmap_{tf}.png"
    plt.savefig(heatmap_path)
    plt.close()

    print(f"📊 Heatmap guardado: {heatmap_path}")

# =========================
# Global CSV
# =========================
df_all = pd.DataFrame(all_results)
global_csv = f"{OUTPUT_DIR}/audit_all_timeframes.csv"
df_all.to_csv(global_csv, index=False)

print(f"\n✅ Auditoría completa")
print(f"📄 CSV global: {global_csv}")
