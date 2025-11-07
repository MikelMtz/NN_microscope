from __future__ import annotations
import os
import numpy as np
import pandas as pd

def load_summary(run_dir: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(run_dir, "kernel", "summary.csv"))

def load_shares(run_dir: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(run_dir, "kernel", "shares.csv"))

def load_top_eigs(run_dir: str, epoch: int):
    e = f"epoch_{epoch:04d}"
    p = os.path.join(run_dir, "kernel", e, "K_L_top_eigvals.npy")
    return np.load(p)