import tkinter as tk
from tkinter import ttk
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# =========================
# CONFIG
# =========================

CSV_PATH = "features/eth_features_1h.csv"
MODEL_PATH = "ml/models/1h/model_regime_1h_h3_t0007.pkl"
FEATURES_PATH = "ml/models/1h/features_regime_1h_h3_t0007.pkl"

WINDOW_SIZE = 50

# =========================
# Load data & model
# =========================

df = pd.read_csv(CSV_PATH)

model = joblib.load(MODEL_PATH)
features_cols = joblib.load(FEATURES_PATH)

X = df[features_cols]
df["pred"] = model.predict(X)

total_windows = len(df) // WINDOW_SIZE
current_window = 0

# =========================
# Tkinter App
# =========================

root = tk.Tk()
root.title("ML Regime Visual Debugger")

fig, ax = plt.subplots(figsize=(10, 4))
ax2 = ax.twinx()

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# =========================
# Plot function
# =========================

def plot_window(idx):
    ax.cla()
    ax2.cla()

    start = idx * WINDOW_SIZE
    end = start + WINDOW_SIZE
    chunk = df.iloc[start:end]

    # Precio
    ax.plot(
        chunk.index,
        chunk["close"],
        label="Close",
        linewidth=2,
        color="tab:blue"
    )
    ax.set_ylabel("ETH Price (USD)")
    ax.grid(True)

    # Regime
    ax2.step(
        chunk.index,
        chunk["pred"],
        where="post",
        label="Regime (0=Short,1=Flat,2=Long)",
        color="tab:orange",
        linewidth=2
    )
    ax2.set_ylabel("Regime")
    ax2.set_ylim(-0.2, 2.2)
    ax2.set_yticks([0, 1, 2])

    ax.set_title(f"Window {idx + 1} / {total_windows}")

    # Leyenda combinada
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    canvas.draw_idle()



# =========================
# Navigation
# =========================

def prev_window():
    global current_window
    if current_window > 0:
        current_window -= 1
        plot_window(current_window)

def next_window():
    global current_window
    if current_window < total_windows - 1:
        current_window += 1
        plot_window(current_window)

def goto_window():
    global current_window
    try:
        val = int(entry.get()) - 1
        if 0 <= val < total_windows:
            current_window = val
            plot_window(current_window)
    except:
        pass
    
def on_close():
    plt.close("all")     # mata matplotlib
    root.quit()          # corta mainloop
    root.destroy()       # destruye la ventana
    
root.protocol("WM_DELETE_WINDOW", on_close)

# =========================
# Controls
# =========================

controls = ttk.Frame(root)
controls.pack(pady=5)

ttk.Button(controls, text="⏮ Prev", command=prev_window).grid(row=0, column=0)
ttk.Button(controls, text="Next ⏭", command=next_window).grid(row=0, column=1)

ttk.Label(controls, text="Go to:").grid(row=0, column=2)
entry = ttk.Entry(controls, width=5)
entry.grid(row=0, column=3)

ttk.Label(controls, text=f"/ {total_windows}").grid(row=0, column=4)
ttk.Button(controls, text="Go", command=goto_window).grid(row=0, column=5)

# =========================
# Init
# =========================

plot_window(current_window)
root.mainloop()
