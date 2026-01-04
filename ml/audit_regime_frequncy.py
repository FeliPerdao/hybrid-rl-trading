import pandas as pd
import joblib
import matplotlib.pyplot as plt

DATA_PATH = "features/eth_features_1h.csv"
MODEL_PATH = "ml/model_regime.pkl"
FEATURES_PATH = "ml/feature_cols_regime.pkl"

def main():
    df = pd.read_csv(DATA_PATH)

    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURES_PATH)

    X = df[feature_cols]
    proba = model.predict_proba(X)[:, 1]

    df["regime_proba"] = proba
    df["regime_on"] = (df["regime_proba"] > 0.7).astype(int)

    hours_per_day = df["regime_on"].sum() / (len(df) / 24)

    print("====== REGIME FREQUENCY ======")
    print(f"Horas operables por día: {hours_per_day:.2f}")

    # --- gráfico ---
    plt.figure(figsize=(10, 4))
    plt.plot(df["regime_proba"], alpha=0.7)
    plt.axhline(0.6, linestyle="--")
    plt.title("Probabilidad de Régimen Operable")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
