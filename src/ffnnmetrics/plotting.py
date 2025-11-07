
from __future__ import annotations
import os
import pandas as pd
import matplotlib.pyplot as plt

COLORS = None  # default styles

def _save(fig, out):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def plot_training_mse(run_dir: str, out_path: str):
    df = pd.read_csv(os.path.join(run_dir, "metrics.csv"))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(df["epoch"], df["train_mse"], label="train")
    ax.plot(df["epoch"], df["test_mse"], label="test")
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE"); ax.legend()
    _save(fig, out_path)


def plot_kernel_scalars(run_dir: str, out_path: str):
    df = pd.read_csv(os.path.join(run_dir, "kernel", "summary.csv"))
    fig = plt.figure(); ax = fig.add_subplot(111)
    for col in ["A_k", "d_eff", "DeltaK", "rho_k", "rho_bar", "S_k", "C", "G"]:
        if col in df.columns:
            ax.plot(df["epoch"], df[col], label=col)
    ax.set_xlabel("epoch"); ax.set_ylabel("value"); ax.legend(ncol=3)
    _save(fig, out_path)


def plot_layer_shares(run_dir: str, out_path: str):
    df = pd.read_csv(os.path.join(run_dir, "kernel", "shares.csv"))
    fig = plt.figure(); ax = fig.add_subplot(111)
    share_cols = [c for c in df.columns if c.startswith("share_layer_")]
    for c in share_cols:
        ax.plot(df["epoch"], df[c], label=c)
    ax.set_xlabel("epoch"); ax.set_ylabel("share"); ax.legend(ncol=2)
    _save(fig, out_path)