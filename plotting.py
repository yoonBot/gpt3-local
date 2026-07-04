"""
Live training-loss line plot.

Works in two contexts:
  - Inside a Jupyter/Colab kernel (train.py imported and run in-process,
    not via `!python train.py` subprocess): redraws in place using an
    IPython display handle, so the chart updates live as training runs.
  - As a plain script/subprocess: no live widget is possible (there's no
    notebook frontend to render into), so it just (re)writes
    {out_dir}/loss_curve.png on every update -- open that file to watch
    progress.

Detection is automatic (`get_ipython()` is only defined inside an IPython
kernel), so the same training code works either way with no flags needed.
"""

import os

import matplotlib

try:
    get_ipython()  # noqa: F821 -- only defined inside an IPython/Colab kernel
    _IN_NOTEBOOK = True
except NameError:
    _IN_NOTEBOOK = False
    matplotlib.use("Agg")  # headless: never try to open a GUI window

import matplotlib.pyplot as plt


class LossPlotter:
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        self.train_iters, self.train_losses = [], []
        self.val_iters, self.val_losses = [], []

        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self._display_handle = None
        if _IN_NOTEBOOK:
            from IPython.display import display
            self._display_handle = display(self.fig, display_id=True)

    def log_train(self, it: int, loss: float):
        self.train_iters.append(it)
        self.train_losses.append(loss)

    def log_val(self, it: int, loss: float):
        self.val_iters.append(it)
        self.val_losses.append(loss)

    def redraw(self):
        self.ax.clear()
        self.ax.plot(self.train_iters, self.train_losses, label="train loss", linewidth=1)
        if self.val_iters:
            self.ax.plot(self.val_iters, self.val_losses, label="val loss", marker="o")
        self.ax.set_xlabel("iteration")
        self.ax.set_ylabel("loss")
        self.ax.set_title("training loss")
        self.ax.legend()
        self.ax.grid(alpha=0.3)

        if _IN_NOTEBOOK:
            self._display_handle.update(self.fig)
        else:
            os.makedirs(self.out_dir, exist_ok=True)
            self.fig.savefig(os.path.join(self.out_dir, "loss_curve.png"))

    def close(self):
        plt.close(self.fig)
