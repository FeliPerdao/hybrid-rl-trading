import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

CSV_PATH = "grid_results.csv"

class HeatmapApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ML Grid Heatmap Viewer")

        self.df = pd.read_csv(CSV_PATH)
        self.df = self.df.sort_values(["horizon", "threshold"])

        self.metrics = [
            "accuracy",
            "f1_macro",
            "f1_weighted",
            "precision_0",
            "recall_0",
            "precision_2",
            "recall_2"
        ]

        self.metric = tk.StringVar(value="f1_macro")

        # =======================
        # UI
        # =======================

        top = tk.Frame(root)
        top.pack(fill=tk.X)

        tk.Label(top, text="Metric:").pack(side=tk.LEFT, padx=5)

        self.combo = ttk.Combobox(top, values=self.metrics, textvariable=self.metric, width=15)
        self.combo.pack(side=tk.LEFT)
        self.combo.bind("<<ComboboxSelected>>", lambda e: self.update_plot())

        btn = tk.Button(top, text="🔄 Refresh", command=self.update_plot)
        btn.pack(side=tk.LEFT, padx=10)

        # =======================
        # Matplotlib Figure
        # =======================

        self.fig, self.ax = plt.subplots(figsize=(12, 7))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.update_plot()

    def update_plot(self):
        metric = self.metric.get()

        self.ax.clear()

        pivot = self.df.pivot(
            index="horizon",
            columns="threshold",
            values=metric
        )

        im = self.ax.imshow(pivot, aspect="auto", origin="lower")

        self.ax.set_xticks(np.arange(len(pivot.columns)))
        self.ax.set_xticklabels([f"{t:.3f}" for t in pivot.columns], rotation=90)

        self.ax.set_yticks(np.arange(len(pivot.index)))
        self.ax.set_yticklabels(pivot.index)

        self.ax.set_xlabel("Threshold")
        self.ax.set_ylabel("Horizon")
        self.ax.set_title(f"Heatmap – {metric}")

        self.fig.colorbar(im, ax=self.ax, label=metric)

        self.fig.tight_layout()
        self.canvas.draw()


def main():
    root = tk.Tk()
    app = HeatmapApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
