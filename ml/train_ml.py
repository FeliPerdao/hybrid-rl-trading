import pandas as pd
import numpy as np
import joblib
import config
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Paths and definition parameters
DATA_PATH = "features/eth_features_1h.csv"
MODEL_PATH = "ml/model_regime.pkl"
FEATURES_PATH = "ml/feature_cols_regime.pkl"

DROP_COLS = [
    "timestamp",
    "open","high","low","close",
    "ema20","ema50","ema100",
    "atr14"
]

def build_target(df):
    H = config.ML_HORIZON
    T = config.ML_TARGET_THRESHOLD
    # Shift close price into the future by HORIZON steps and compute future return
    future_close = df["close"].shift(-H)
    future_ret = (future_close / df["close"]) - 1
    
    #The target is builted with the bias detection up, down or noise (in betweeen)
    #It makes neutral (1) in all table, and later modifies to down (0) or up (2) when corresponds
    target = np.ones(len(df), dtype=int)   # 1 = neutro
    target[future_ret < -T] = 0
    target[future_ret > T] = 2

    return pd.Series(target, index=df.index)


def main(return_report=False):
    df = pd.read_csv(DATA_PATH)

    df["target"] = build_target(df)
    df = df.dropna()

    features = df.drop(columns=DROP_COLS + ["target"])
    target = df["target"]
    joblib.dump(features.columns.tolist(), FEATURES_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.25,
        shuffle=False
    )
    
    classes = np.array([0,1,2])
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))

    # Gradient Boosting classifier for regime detection
    model = GradientBoostingClassifier(
        n_estimators=300,   #number of boosting stages
        learning_rate=0.05, #shrinkage to reduce overfitting
        max_depth=3,        #tree complexity
        random_state=42
    )

    #Traing
    model.fit(X_train, y_train, sample_weight=y_train.map(class_weights))

    #Predict regime on unseen future data
    preds = model.predict(X_test)

    #Diagnostics
    report = classification_report(y_test, preds, digits=3, output_dict=True)
    print(f"ML target threshold: {config.ML_TARGET_THRESHOLD}")
    print(f"ML horizon: {config.ML_HORIZON}")

    print("====== REGIME FILTER AUDIT ======")
    print(classification_report(y_test, preds, digits=3))

    joblib.dump(model, MODEL_PATH)
    print(f"✔ Modelo guardado en {MODEL_PATH}")
    
    if return_report:
        return report


def run():
    main()
