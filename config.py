# --- parámetros sensibles ---
REGIME_THRESHOLD = 0.7
PENALTY_FACTOR = 0.15
ML_TARGET = "return_5m"
ML_HORIZON = 24  # number of candles for classification
ML_TARGET_THRESHOLD = 0.04  # Variation threshold after "HORIZON" candles to classify tradeable regime
MAX_HOLD = 24 #max candles it can wait to close operation
TIMEFRAME = "1h"
ASSET = "ETH"