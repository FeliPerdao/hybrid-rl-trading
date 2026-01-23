from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env_trading_rl import TradingEnvRL
import pandas as pd

df = pd.read_csv("regime_classified/1h/eth_regime_1h_h3_t0007.csv")

def make_env():
    return TradingEnvRL(df)

env = DummyVecEnv([make_env])
env = VecNormalize(
    env,
    norm_obs=True,
    norm_reward=True,
    clip_obs=10.0
)

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=lambda f: 3e-4 * f,   # decay
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01,
    clip_range=0.2,
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=1
)

model.learn(total_timesteps=1_000_000)

model.save("models/ppo_trading_v2")
env.save("models/vec_normalize.pkl")