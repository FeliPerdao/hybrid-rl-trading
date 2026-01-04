import os
import json
from datetime import datetime

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "backtests.jsonl")


def log_backtest_result(
    final_equity,
    total_pnl,
    trades,
    win_rate,
    profit_factor,
    max_dd,
    params: dict
):
    os.makedirs(LOG_DIR, exist_ok=True)

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "final_equity": round(float(final_equity), 6),
        "total_pnl": round(float(total_pnl), 6),
        "trades": int(trades),
        "win_rate": round(float(win_rate), 4) if win_rate is not None else None,
        "profit_factor": round(float(profit_factor), 4) if profit_factor is not None else None,
        "max_drawdown": round(float(max_dd), 6),
        "params": params
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    print(f"📝 Backtest logueado en {LOG_FILE}")
