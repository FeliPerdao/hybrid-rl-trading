# ml/grid_backtest_ml.py

import pandas as pd
from datetime import datetime
import time

from ml.x_backtest_ml import backtest_one

# ==============================
# Grid definition
# ==============================

TIMEFRAME = "30m"

HORIZONS = [5, 6, 7, 8]
THRESHOLDS = [0.01, 0.012, 0.014, 0.016]

# ==============================
# Run grid
# ==============================

def main():
    t0 = time.time()
    results = []

    for h in HORIZONS:
        for t in THRESHOLDS:
            print(f"\n🧪 Backtest ML | TF={TIMEFRAME} H={h} T={t}")
            try:
                res = backtest_one(TIMEFRAME, h, t)
                results.append(res)
            except Exception as e:
                print("❌ Error:", e)

    df = pd.DataFrame(results)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"ml_backtest_grid_{TIMEFRAME}_{ts}.csv"
    df.to_csv(path, index=False)

    elapsed = time.time() - t0

    print("\n✅ Grid backtest ML terminado")
    print(f"📄 {path}")
    print(f"⏱ Tiempo total: {elapsed/60:.2f} min")

if __name__ == "__main__":
    main()
