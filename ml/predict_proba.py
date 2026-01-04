import pandas as pd
import joblib

DATA_PATH = "features/eth_features_1h.csv"

MODEL_REGIME = "ml/model_regime.pkl"
FEATURES_REGIME = "ml/feature_cols_regime.pkl"

MODEL_DIR = "ml/model_gb.pkl"
FEATURES_DIR = "ml/feature_cols.pkl"

REGIME_THRESHOLD = 0.7


def main():
    df = pd.read_csv(DATA_PATH)

    # === REGIME ===
    model_regime = joblib.load(MODEL_REGIME)
    feat_regime = joblib.load(FEATURES_REGIME)

    Xr = df[feat_regime].iloc[[-1]]
    p_regime = model_regime.predict_proba(Xr)[0][1]

    print("====== REGIME CHECK ======")
    print(f"Probabilidad régimen operable: {p_regime:.2%}")

    if p_regime < REGIME_THRESHOLD:
        print("⛔ Mercado NO operable. No hacer nada.")
        return

    print("✅ Mercado operable. Evaluando dirección...")

    # === DIRECCIÓN ===
    model_dir = joblib.load(MODEL_DIR)
    feat_dir = joblib.load(FEATURES_DIR)

    Xd = df[feat_dir].iloc[[-1]]
    proba = model_dir.predict_proba(Xd)[0]
    classes = model_dir.classes_

    proba_map = dict(zip(classes, proba))

    p_up = proba_map.get(1, 0.0)
    p_down = proba_map.get(-1, 0.0)
    edge = p_up - p_down

    print("====== DIRECTION ======")
    print(f"📈 P(up):   {p_up:.2%}")
    print(f"📉 P(down): {p_down:.2%}")
    print(f"⚖️ Edge:   {edge:.2%}")

    if edge > 0.15:
        print("🟢 Bias LONG")
    elif edge < -0.15:
        print("🔴 Bias SHORT")
    else:
        print("⚪ Sin bias claro")


if __name__ == "__main__":
    main()
