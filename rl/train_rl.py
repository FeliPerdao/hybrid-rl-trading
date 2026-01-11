import pandas as pd
import joblib
import config
from stable_baselines3 import DQN
from rl.env_trading import TradingEnv

DATA_PATH = "features/eth_features_1h.csv"

MODEL_REGIME = "ml/model_regime.pkl"
FEATURES_REGIME = "ml/feature_cols_regime.pkl"

RL_MODEL_PATH = "rl/dqn_trader_regime"

#REGIME_THRESHOLD = config.REGIME_THRESHOLD
#MAX_HOLD = config.MAX_HOLD


def main():
    df = pd.read_csv(DATA_PATH)

    feature_cols = joblib.load(FEATURES_REGIME)
    features = df[feature_cols]
    prices = df["close"]

    model_regime = joblib.load(MODEL_REGIME)
    regime_proba = model_regime.predict_proba(features)[:, 1]
    regime_on = (regime_proba > config.REGIME_THRESHOLD).astype(int)


    env = TradingEnv(
        features_df=features,
        price_series=prices,
        regime_on=pd.Series(regime_on),
        max_hold=config.MAX_HOLD,
    )

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-4,
        buffer_size=50_000,
        batch_size=64,
        gamma=0.99,
        exploration_fraction=0.15,
        exploration_final_eps=0.05,
        verbose=1
    )

    model.learn(total_timesteps=150_000)
    model.save(RL_MODEL_PATH)

    print("✔ RL entrenado con filtro de régimen")



def run():
    main()
