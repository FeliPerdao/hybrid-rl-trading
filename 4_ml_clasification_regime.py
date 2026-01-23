import os
import glob
import joblib
import pandas as pd

# =========================
# PATHS
# =========================

FEATURES_BASE = "features"
MODELS_BASE   = "ml/models"
OUTPUT_BASE   = "regime_classified"

# =========================
# Utils
# =========================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# =========================
# Main
# =========================

for tf_dir in os.listdir(MODELS_BASE):
    tf_path = os.path.join(MODELS_BASE, tf_dir)

    if not os.path.isdir(tf_path):
        continue

    print(f"\n=== TF {tf_dir} ===")

    # CSV base (features originales)
    csv_path = os.path.join(FEATURES_BASE, f"eth_features_{tf_dir}.csv")
    if not os.path.exists(csv_path):
        print(f"❌ No existe {csv_path}")
        continue

    df = pd.read_csv(csv_path)

    model_files = glob.glob(os.path.join(tf_path, "model_regime_*.pkl"))

    for model_path in model_files:
        name = os.path.basename(model_path).replace(".pkl", "")
        feat_path = model_path.replace("model_", "features_")

        print(f"→ Procesando {name}")

        model = joblib.load(model_path)
        features = joblib.load(feat_path)

        # sanity
        missing = set(features) - set(df.columns)
        if missing:
            print(f"❌ Faltan features: {missing}")
            continue

        X = df[features]
        regime = model.predict(X)

        df_out = df.copy()
        df_out["regime"] = regime

        # output
        out_dir = os.path.join(OUTPUT_BASE, tf_dir)
        ensure_dir(out_dir)

        out_file = name.replace("model_", "eth_") + ".csv"
        out_path = os.path.join(out_dir, out_file)

        df_out.to_csv(out_path, index=False)
        print(f"✅ Guardado {out_path}")
