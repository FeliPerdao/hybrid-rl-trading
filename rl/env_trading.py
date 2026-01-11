import gym
import numpy as np

class TradingEnv(gym.Env):
    TRADE_FEE = 0.0033 #Fixed fee applied on every close
    def set_mode(self, mode: str):
        # "train" -> exploration + penalties
        # "backtest" -> realistic execution
        self.mode = mode

    def __init__(self, features_df, price_series, regime_on, max_hold=24):
        super().__init__()

        #Market features
        self.X = features_df.reset_index(drop=True)
        
        #Price used to calculate PnL
        self.prices = price_series.reset_index(drop=True)
        
        #Regime filter (1 = tradable / 0 = no-trade)
        self.regime_on = regime_on.reset_index(drop=True)

        self.n_features = self.X.shape[1]
        self.max_hold = max_hold
        self.mode = "train"

        #Actions: 0 -> HOLD / 1 -> OPEN / 2 -> CLOSE
        self.action_space = gym.spaces.Discrete(3)
        
        #Observation
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_features + 3,),
            dtype=np.float32
        )

        self.reset()

    def reset(self, start_idx=0):
        #Random start for training to avoid overfitting to one period
        if self.mode == "train":
            self.step_idx = np.random.randint(0, len(self.X) - 500)
        else:
            self.step_idx = start_idx
            
        #Position state
        self.position = 0
        self.entry_price = 0.0
        self.hold_time = 0
        
        return self._get_obs()



    def _get_obs(self):
        #Build observation vector
        return np.concatenate([
            self.X.iloc[self.step_idx].values.astype(np.float32),
            np.array([
                float(self.regime_on.iloc[self.step_idx]),
                self.position,
                self.hold_time
            ],
            dtype=np.float32)
        ])

    def step(self, action):
        pnl = 0.0
        reward = 0.0
        done = False
        trade_hold_time = None

        price = self.prices.iloc[self.step_idx]
        regime = float(self.regime_on.iloc[self.step_idx])
        regime_weight = regime
        
        #Rule: forbid closing a position in the same candle (prevents zero-duration trades)
        if action == 2 and self.position == 1 and self.hold_time == 0:
            action = 0

        # THESE RULES BELOW WHERE MODIFIED TO CHANGE HARD RULES FOR SOFT WEIGHTS
        # #Backtest rule: oppening trades is forbiden when regime is OFF
        # if self.mode == "backtest" and regime == 0 and action == 1:
        #     action = 0

        # #Training rule: penalize trying to trade in no trade regime
        # if self.mode == "train" and regime == 0 and action != 0:
        #     reward -= 0.005

        #ACTION: OPEN LONG -> regime is ON & agent is flat
        if regime == 1 and action == 1 and self.position == 0:
            self.position = 1
            self.entry_price = price
            
        #ACTION: CLOSE (normal exit) 
        if action == 2 and self.position == 1:
            pnl = (price - self.entry_price) / self.entry_price
            trade_hold_time = self.hold_time
            
            #Base reward: PnL - trading fee
            reward += (pnl * regime_weight) - self.TRADE_FEE
            
            #Bonus for profitable trades and penalty for losses
            reward += (
                0.004 if pnl > 0 
                else -0.01 if pnl < -0.02 
                else 0.0
            )
            
            self.position = 0
            self.hold_time = 0
            
        #Holding logic
        if self.position == 1:
            self.hold_time += 1
            #Penalty to discourage over-holding
            reward -= 0.001 * (1.0 + (1.0 - regime_weight))
            
            #Time-out exit (force close if max holding time exceeded)
            if self.hold_time > self.max_hold:
                pnl = (price - self.entry_price) / self.entry_price
                trade_hold_time = self.hold_time
                reward += pnl - 0.002 - self.TRADE_FEE
                self.position = 0
                self.hold_time = 0

        #Advance environment
        self.step_idx += 1
        done = self.step_idx >= len(self.X) - 1

        return self._get_obs(), reward, done, {
            "pnl": pnl,
            "hold_time": trade_hold_time if pnl != 0 else None
    }