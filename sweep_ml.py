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

# ==============================
# Hook train_ml to return metrics
# ==============================

def run_one(horizon, threshold):
    config.ML_HORIZON = horizon
    config.ML_TARGET_THRESHOLD = threshold

    print(f"\n🧪 Running: H={horizon}  T={threshold}")

    report = train_ml.main(return_report=True)

    return {
        "horizon": horizon,
        "threshold": threshold,
        "accuracy": report["accuracy"],
        "f1_macro": report["macro avg"]["f1-score"],
        "f1_weighted": report["weighted avg"]["f1-score"],
        "precision_0": report["0"]["precision"],
        "recall_0": report["0"]["recall"],
        "precision_2": report["2"]["precision"],
        "recall_2": report["2"]["recall"]
    }

# ==============================
# Main sweep loop
# ==============================

def main():
    t0 = time.time()
    for h in HORIZONS:
        for t in THRESHOLDS:
            try:
                res = run_one(h, t)
                RESULTS.append(res)
            except Exception as e:
                print("❌ Error:", e)

    df = pd.DataFrame(RESULTS)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"ml_sweep_{ts}.csv"
    json_path = f"ml_sweep_{ts}.json"

    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)
    
    elapsed = time.time() - t0

    print("\n✅ Sweep complete")
    print(f"📄 {csv_path}")
    print(f"📄 {json_path}") 

    print(f"⏱ Tiempo total: {elapsed/60:.2f} minutos ({elapsed:.1f} segundos)")

if __name__ == "__main__":
    main()
