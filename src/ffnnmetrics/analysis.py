from __future__ import annotations
import os
import numpy as np
import pandas as pd

def load_summary(run_dir: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(run_dir, "kernel", "summary.csv"))

def load_shares(run_dir: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(run_dir, "kernel", "shares.csv"))

def load_top_eigs(run_dir: str, epoch: int):
    return np.load(os.path.join(run_dir, "kernel", f"K_L_top_eigvals_epoch{epoch:04d}.npy"))

# NEW
def load_lastlayer_advanced(run_dir: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(run_dir, "kernel", "lastlayer_advanced.csv"))

def load_interlayer_ilard(run_dir: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(run_dir, "kernel", "interlayer_ilard.csv"))

def load_interlayer_depth_profiles(run_dir: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(run_dir, "kernel", "interlayer_depth_profiles.csv"))

def load_transported_alignment(run_dir: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(run_dir, "kernel", "transported_alignment.csv"))
