import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = "features/eth_features_1h.csv"
MODEL_PATH = "ml/model_gb.pkl"
FEATURES_PATH = "ml/feature_cols.pkl"

EDGE_THRESHOLD = 0.15  # lo que consideramos señal real


def main():
    df = pd.read_csv(DATA_PATH)

    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURES_PATH)

    X = df[feature_cols]
    probs = model.predict_proba(X)
    classes = model.classes_

    proba_df = pd.DataFrame(probs, columns=classes)
    proba_df["edge"] = proba_df.get(1, 0) - proba_df.get(-1, 0)

    # --- STATS ---
    strong = proba_df[np.abs(proba_df["edge"]) > EDGE_THRESHOLD]

    signals_per_day = len(strong) / (len(df) / 24)

    print("====== EDGE AUDIT ======")
    print(f"Total velas: {len(df)}")
    print(f"Señales fuertes: {len(strong)}")
    print(f"Señales por día: {signals_per_day:.2f}")
    print(f"Edge medio: {proba_df['edge'].mean():.4f}")
    print(f"Edge |95%|: {np.percentile(np.abs(proba_df['edge']), 95):.3f}")

    # --- HISTOGRAMA ---
    plt.figure(figsize=(10, 4))
    plt.hist(proba_df["edge"], bins=100)
    plt.axvline(EDGE_THRESHOLD, linestyle="--")
    plt.axvline(-EDGE_THRESHOLD, linestyle="--")
    plt.title("Distribución del Edge (P_up - P_down)")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
