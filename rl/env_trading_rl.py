#sugerencias del chatgpt pendientes
# ❌ No toqué regime
# ❌ No toqué hold
# ❌ No toqué reset ni VecNormalize

import gym
import numpy as np


class TradingEnvRL(gym.Env):
    def __init__(self, df, initial_balance=10_000, fee=0.001, eval_mode=False):
        super().__init__()
        self.eval_mode = eval_mode

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.fee = fee  # 0.1%

        # 0 = HOLD, 1 = OPEN, 2 = CLOSE
        self.action_space = gym.spaces.Discrete(3)

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(7,),
            dtype=np.float32
        )

        self.reset()

    def reset(self):
        if self.eval_mode:
            self.step_idx = 10
        else:
            self.step_idx = np.random.randint(10, len(self.df) - 200)

        self.balance = self.initial_balance
        self.position = 0
        self.position_value = 0.0
        self.entry_price = 0.0
        self.time_in_pos = 0

        return self._get_obs()

    def _get_obs(self):
        row = self.df.iloc[self.step_idx]

        unrealized = (
            (row["close"] - self.entry_price) / self.entry_price
            if self.position else 0.0
        )

        return np.array([
            row["ret_1"],
            row["ret_4"],
            row["atr_pct"],
            self.position,
            unrealized,
            self.time_in_pos / 100.0,
            row["regime"]
        ], dtype=np.float32)

    def step(self, action):
        done = False
        reward = 0.0

        row = self.df.iloc[self.step_idx]
        price = row["close"]
        regime = row["regime"]

        # ======================
        # OPEN
        # ======================
        if action == 1 and self.position == 0:
            self.position = 1
            
            
            self.entry_price = price * (1 + self.fee)
            self.position_value = self.balance
            self.balance = 0.0
            self.time_in_pos = 0
            
            reward -= self.fee * 10
            if regime != 2:
                reward -= 0.2


        # ======================
        # CLOSE
        # ======================
        elif action == 2 and self.position == 1:
            exit_price = price * (1 - self.fee)
            
            pnl = (exit_price - self.entry_price) / self.entry_price

            self.balance = self.position_value * (1 + pnl)
            
            reward += pnl * 50
            reward -= self.fee * 10

            self.position = 0
            self.position_value = 0.0
            self.entry_price = 0.0
            self.time_in_pos = 0

        # ======================
        # HOLD COST
        # ======================
        if self.position:
            self.time_in_pos += 1
            
            MAX_HOLD = 12
            if self.time_in_pos > MAX_HOLD:
                reward -= 0.001*self.time_in_pos
                
        # # ====================== 
        # # HOLD COST viejo
        # # ====================== 
        # if self.position:
        #     self.time_in_pos += 1
        #     reward -= 0.005
        #     reward -= 0.001 * self.time_in_pos

        # ======================
        # STEP
        # ======================
        self.step_idx += 1
        
        if self.step_idx >= len(self.df) - 1:
            if self.position == 1:
                final_price = self.df.iloc[self.step_idx]["close"]
                exit_price = final_price * (1 - self.fee)

                pnl = (exit_price - self.entry_price) / self.entry_price
                self.balance = self.position_value * (1 + pnl)
                
                #Final reward at dataset ending
                reward += pnl * 50
                reward -= self.fee * 10
            done = True
            
        return self._get_obs(), reward, done, {}
