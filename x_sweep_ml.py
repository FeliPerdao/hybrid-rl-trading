import itertools
import json
import pandas as pd
import numpy as np
from datetime import datetime
import time

import config
import ml.train_ml as train_ml

from sklearn.metrics import classification_report

# ==============================
# Timeframe selector
# ==============================

def select_timeframe():
    print("\nSeleccioná el timeframe para el sweep ML:\n")

    for i, tf in enumerate(train_ml.AVAILABLE_TFS, 1):
        print(f"{i} - {tf}")

    try:
        sel = int(input("\nElegí UNO: ").strip())
        return train_ml.AVAILABLE_TFS[sel - 1]
    except:
        print("❌ Selección inválida.")
        exit(1)


TF = select_timeframe()
print(f"\n🧠 Sweep sobre timeframe: {TF}")

# ==============================
# Sweep definitions
# ==============================

def build_horizons():
    h = []
    h += list(range(3, 13, 1))     # 3 → 12 step 1
    h += list(range(14, 25, 2))    # 14 → 24 step 2
    h += list(range(28, 49, 4))    # 28 → 48 step 4
    return h

HORIZONS = build_horizons()
THRESHOLDS = np.round(np.arange(0.01, 0.0401, 0.002), 3)

RESULTS = []
CSV_PATH = None
JSON_PATH = None

# ==============================
# Hook train_ml to return metrics
# ==============================

def run_one(horizon, threshold):
    config.ML_HORIZON = horizon
    config.ML_TARGET_THRESHOLD = threshold

    print(f"\n🧪 Running: TF={TF}  H={horizon}  T={threshold}")

    out = train_ml.main(timeframes=[TF], return_report=True)
    report = out[TF]

    return {
        "timeframe": TF,
        "horizon": horizon,
        "threshold": threshold,

        "accuracy": float(report["accuracy"]),
        "f1_macro": float(report["macro avg"]["f1-score"]),
        "f1_weighted": float(report["weighted avg"]["f1-score"]),

        "precision_0": float(report["0"]["precision"]),
        "recall_0": float(report["0"]["recall"]),
        "precision_2": float(report["2"]["precision"]),
        "recall_2": float(report["2"]["recall"]),
    }


# ==============================
# Main sweep loop
# ==============================

def main():
    global CSV_PATH, JSON_PATH

    t0 = time.time()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    CSV_PATH = f"ml_sweep_{TF}_{ts}.csv"
    JSON_PATH = f"ml_sweep_{TF}_{ts}.json"

    # crear archivos vacíos con headers
    pd.DataFrame(columns=[
        "timeframe","horizon","threshold",
        "accuracy","f1_macro","f1_weighted",
        "precision_0","recall_0","precision_2","recall_2"
    ]).to_csv(CSV_PATH, index=False)

    with open(JSON_PATH, "w") as f:
        json.dump([], f)

    print(f"📄 Guardando en {CSV_PATH}")

    # === LOOP ===
    for h in HORIZONS:
        for t in THRESHOLDS:
            try:
                res = run_one(h, t)
                RESULTS.append(res)

                # append CSV
                pd.DataFrame([res]).to_csv(
                    CSV_PATH, mode="a", header=False, index=False
                )

                # append JSON
                with open(JSON_PATH, "r+") as f:
                    data = json.load(f)
                    data.append(res)
                    f.seek(0)
                    json.dump(data, f, indent=2)

            except Exception as e:
                print("❌ Error:", e)

    elapsed = time.time() - t0

    print("\n✅ Sweep complete")
    print(f"📄 {CSV_PATH}")
    print(f"📄 {JSON_PATH}")
    print(f"⏱ Tiempo total: {elapsed/60:.2f} minutos ({elapsed:.1f} segundos)")

if __name__ == "__main__":
    main()
