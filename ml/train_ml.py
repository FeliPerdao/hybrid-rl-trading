import pandas as pd
import joblib
import config
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Paths and definition parameters
DATA_PATH = "features/eth_features_1h.csv"
MODEL_PATH = "ml/model_regime.pkl"
FEATURES_PATH = "ml/feature_cols_regime.pkl"

HORIZON = config.ML_HORIZON
THRESHOLD = config.ML_TARGET_THRESHOLD

def build_target(df):
    # Shift close price into the future by HORIZON steps
    future_close = df["close"].shift(-HORIZON)
    
    #Compute future return over the horizon
    future_ret = (future_close / df["close"]) - 1
    
    #Regime label: 1-> price movement (tradable regime) / 0 -> low volatility or noise (no-trade regime)
    return (future_ret.abs() > THRESHOLD).astype(int)


def main():
    df = pd.read_csv(DATA_PATH)

    df["target"] = build_target(df)
    df = df.dropna()

    features = df.drop(columns=["timestamp", "target"])
    target = df["target"]

    joblib.dump(features.columns.tolist(), FEATURES_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.25,
        shuffle=False
    )

    # Gradient Boosting classifier for regime detection
    model = GradientBoostingClassifier(
        n_estimators=300,   #number of boosting stages
        learning_rate=0.05, #shrinkage to reduce overfitting
        max_depth=3,        #tree complexity
        random_state=42
    )

    #Traing
    model.fit(X_train, y_train)

    #Predict regime on unseen future data
    preds = model.predict(X_test)

    #Diagnostics
    print(f"ML target threshold: {THRESHOLD}")
    print(f"ML horizon: {HORIZON}")

    print("====== REGIME FILTER AUDIT ======")
    print(classification_report(y_test, preds, digits=3))

    joblib.dump(model, MODEL_PATH)
    print(f"✔ Modelo guardado en {MODEL_PATH}")


def run():
    main()
