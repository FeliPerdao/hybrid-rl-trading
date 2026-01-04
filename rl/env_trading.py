import gym
import numpy as np

class TradingEnv(gym.Env):
    def set_mode(self, mode: str):
        self.mode = mode  # "train" | "backtest"

    def __init__(self, features_df, price_series, regime_on, max_hold=24):
        super().__init__()

        self.X = features_df.reset_index(drop=True)
        self.prices = price_series.reset_index(drop=True)
        self.regime_on = regime_on.reset_index(drop=True)

        self.n_features = self.X.shape[1]
        self.max_hold = max_hold
        self.mode = "train"

        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_features + 3,),
            dtype=np.float32
        )

        self.reset()

    def reset(self, start_idx=0):
        if self.mode == "train":
            self.step_idx = np.random.randint(0, len(self.X) - 500)
        else:
            self.step_idx = start_idx
        self.position = 0
        self.entry_price = 0.0
        self.hold_time = 0
        return self._get_obs()



    def _get_obs(self):
        obs = np.concatenate([
            self.X.iloc[self.step_idx].values.astype(np.float32),
            np.array([
                self.regime_on.iloc[self.step_idx],
                self.position,
                self.hold_time
            ], dtype=np.float32)
        ])
        return obs

    def step(self, action):
        reward = 0.0
        pnl = 0.0
        done = False

        price = self.prices.iloc[self.step_idx]
        regime = self.regime_on.iloc[self.step_idx]

        if self.mode == "train" and regime == 0 and action != 0:
            reward -= 0.005

        if regime == 1 and action == 1 and self.position == 0:
            self.position = 1
            self.entry_price = price

        if action == 2 and self.position == 1:
            pnl = (price - self.entry_price) / self.entry_price
            reward += pnl
            if pnl < -0.02:
                reward -= 0.01
            if pnl > 0:
                reward += 0.004

            self.position = 0
            self.hold_time = 0

        if self.position == 1:
            self.hold_time += 1
            reward -= 0.0002

            if self.mode == "train" and self.hold_time > self.max_hold:
                reward -= 0.002

        self.step_idx += 1
        if self.step_idx >= len(self.X) - 1:
            done = True

        return self._get_obs(), reward, done, {"pnl": pnl}
