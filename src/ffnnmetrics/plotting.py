from __future__ import annotations
import os, re, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

COLORS = None  # use default style

def _save(fig, out):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)

def _epochs_from_summary(run_dir: str) -> np.ndarray:
    path = os.path.join(run_dir, "kernel", "summary.csv")
    if not os.path.exists(path):
        return np.array([], dtype=int)
    df = pd.read_csv(path)
    return df["epoch"].to_numpy(dtype=int)

def _num_layers_from_rot_csv(run_dir: str) -> int | None:
    p = os.path.join(run_dir, "kernel", "rotation_act.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    cols = [c for c in df.columns if c.startswith("rho_act_layer_")]
    if not cols:
        return None
    # layers are 1..L; return L
    return max(int(c.split("_")[-1]) for c in cols)

def _available_layers_from_files(run_dir: str, which: str) -> list[int]:
    """
    which: 'act' -> looks for C_act_eigvals_layer{ell}_epoch*.npy (ell=0..L)
           'bp'  -> looks for C_bp_eigvals_layer{ell}_epoch*.npy  (ell=1..L)
    """
    pat = "C_act_eigvals_layer*_epoch*.npy" if which == "act" else "C_bp_eigvals_layer*_epoch*.npy"
    files = glob.glob(os.path.join(run_dir, "kernel", pat))
    layers = set()
    for f in files:
        m = re.search(r"layer(\d+)_epoch", os.path.basename(f))
        if m: layers.add(int(m.group(1)))
    return sorted(layers)

# ---------------- basic plots ----------------

def plot_training_mse(run_dir: str, out_path: str):
    p = os.path.join(run_dir, "metrics.csv")
    if not os.path.exists(p):
        print(f"[plot_training_mse] missing {p}")
        return
    df = pd.read_csv(p)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(df["epoch"], df["train_mse"], label="train")
    ax.plot(df["epoch"], df["test_mse"], label="test")
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE"); ax.legend()
    _save(fig, out_path)

def plot_kernel_scalars(run_dir: str, out_path: str):
    p = os.path.join(run_dir, "kernel", "summary.csv")
    if not os.path.exists(p):
        print(f"[plot_kernel_scalars] missing {p}")
        return
    df = pd.read_csv(p)
    fig = plt.figure(); ax = fig.add_subplot(111)
    # feature-space scalars (DeltaC instead of DeltaK)
    for col in ["A_k", "d_eff", "DeltaC", "rho_k", "rho_bar", "S_k", "C", "G"]:
        if col in df.columns:
            ax.plot(df["epoch"], df[col], label=col)
    ax.set_xlabel("epoch"); ax.set_ylabel("value"); ax.legend(ncol=3)
    _save(fig, out_path)

def plot_layer_shares(run_dir: str, out_path: str):
    p = os.path.join(run_dir, "kernel", "shares.csv")
    if not os.path.exists(p):
        print(f"[plot_layer_shares] missing {p}")
        return
    df = pd.read_csv(p)
    fig = plt.figure(); ax = fig.add_subplot(111)
    share_cols = [c for c in df.columns if c.startswith("share_layer_")]
    for c in share_cols:
        ax.plot(df["epoch"], df[c], label=c)
    ax.set_xlabel("epoch"); ax.set_ylabel("share"); ax.legend(ncol=2)
    _save(fig, out_path)

# --------------- new plots ----------------

def plot_rotation_per_layer(run_dir: str, out_path: str, which: str = "act", logx: bool = True, logy: bool = True):
    """
    which: 'act' or 'bp'
    Plots all layers' rotation scores in one figure.
    """
    fname = "rotation_act.csv" if which == "act" else "rotation_bp.csv"
    p = os.path.join(run_dir, "kernel", fname)
    if not os.path.exists(p):
        print(f"[plot_rotation_per_layer] missing {p}")
        return
    df = pd.read_csv(p)
    fig = plt.figure(); ax = fig.add_subplot(111)
    x = df["epoch"].to_numpy()
    x_plot = x + 1 if logx else x  # avoid log(0)
    cols = [c for c in df.columns if c.startswith(f"rho_{which}_layer_")]
    for c in cols:
        y = df[c].to_numpy()
        ax.plot(x_plot, y, label=c)
    if logx: ax.set_xscale("log")
    if logy: ax.set_yscale("log")
    ax.set_xlabel("epoch" + (" (log)" if logx else ""))
    ax.set_ylabel("rotation ρ_k" + (" (log)" if logy else ""))
    ax.legend(ncol=2)
    _save(fig, out_path)

def plot_alignment_per_layer(run_dir: str, out_path: str, logx: bool = True):
    """
    Plots per-layer alignment (δ̄_e alignment to top-k of C_act^(ℓ)) in one figure.
    """
    p = os.path.join(run_dir, "kernel", "alignment.csv")
    if not os.path.exists(p):
        print(f"[plot_alignment_per_layer] missing {p}")
        return
    df = pd.read_csv(p)
    fig = plt.figure(); ax = fig.add_subplot(111)
    x = df["epoch"].to_numpy()
    x_plot = x + 1 if logx else x
    cols = [c for c in df.columns if c.startswith("align_layer_")]
    for c in cols:
        y = df[c].to_numpy()
        ax.plot(x_plot, y, label=c)
    if logx: ax.set_xscale("log")
    ax.set_xlabel("epoch" + (" (log)" if logx else ""))
    ax.set_ylabel("alignment")
    ax.legend(ncol=2)
    _save(fig, out_path)

def plot_transfer_inflow_topk_per_layer(run_dir: str, out_dir: str):
    """
    Loads transfer_inflow_topk_epochXXXX.npy (shape [L, k]) across epochs and
    plots, for each layer ℓ=1..L, the inflow into last-layer top-k modes:
    lines are modes i=1..k with a continuous colormap; x axis is epoch (log scale).
    """
    epochs = _epochs_from_summary(run_dir)
    if epochs.size == 0:
        print("[plot_transfer_inflow_topk_per_layer] no summary.csv to infer epochs")
        return

    # Load all available inflow files, stack as [E, L, K]
    inflows = []
    valid_epochs = []
    for e in epochs:
        p = os.path.join(run_dir, "kernel", f"transfer_inflow_topk_epoch{e:04d}.npy")
        if os.path.exists(p):
            arr = np.load(p)  # [L, k]
            inflows.append(arr)
            valid_epochs.append(e)
    if not inflows:
        print("[plot_transfer_inflow_topk_per_layer] no transfer_inflow files found")
        return

    inflows = np.stack(inflows, axis=0)  # [E, L, K]
    epochs_v = np.array(valid_epochs, dtype=int)
    # Determine min-K across epochs (safety)
    K = min(a.shape[1] for a in inflows)
    inflows = inflows[:, :, :K]

    L = inflows.shape[1]
    x = epochs_v + 1  # log-safe

    cmap = cm.get_cmap("viridis", K)
    for ell in range(1, L + 1):
        fig = plt.figure(); ax = fig.add_subplot(111)
        for i in range(K):
            y = inflows[:, ell-1, i]
            ax.plot(x, y, color=cmap(i), label=f"mode_{i+1}")
        ax.set_xscale("log")
        ax.set_xlabel("epoch (log)")
        ax.set_ylabel(r"$\sum_p T^{(\ell)}_{p\to i}$")
        ax.set_title(f"Layer {ell} transfer inflow to top-k modes")
        # Optional legend (can get busy); show a small colorbar instead:
        # ax.legend(ncol=2, fontsize=8)
        sm = cm.ScalarMappable(cmap=cmap)
        sm.set_array(np.arange(1, K+1))
        cbar = plt.colorbar(sm, ax=ax, pad=0.01)
        cbar.set_label("mode index")
        _save(fig, os.path.join(out_dir, f"transfer_inflow_layer{ell}.png"))

def plot_eigenvalues_over_epochs_per_layer(run_dir: str, out_dir: str, which: str = "act"):
    """
    For each layer, plot the top-k eigenvalues vs epoch (y log scale),
    separate figure per layer, continuous colormap over the mode index.
    which='act' (forward/activations, layers 0..L) or 'bp' (backprop, layers 1..L).
    """
    epochs = _epochs_from_summary(run_dir)
    if epochs.size == 0:
        print("[plot_eigenvalues_over_epochs_per_layer] no summary.csv to infer epochs")
        return

    layers = _available_layers_from_files(run_dir, which=which)
    if not layers:
        print(f"[plot_eigenvalues_over_epochs_per_layer] no eigenvalue files for which='{which}'")
        return

    for ell in layers:
        # Gather eig vals across epochs for this layer
        vals_list = []
        valid_epochs = []
        for e in epochs:
            fn = f"C_act_eigvals_layer{ell}_epoch{e:04d}.npy" if which == "act" else f"C_bp_eigvals_layer{ell}_epoch{e:04d}.npy"
            p = os.path.join(run_dir, "kernel", fn)
            if os.path.exists(p):
                vals = np.load(p)  # [k_ell]
                vals_list.append(vals)
                valid_epochs.append(e)
        if not vals_list:
            continue

        # Stack as [E, K]; trim to min-K across epochs
        minK = min(v.shape[0] for v in vals_list)
        Y = np.stack([v[:minK] for v in vals_list], axis=0)  # [E, minK]
        X = np.array(valid_epochs, dtype=int) + 1  # log-safe

        cmap = cm.get_cmap("viridis", minK)
        fig = plt.figure(); ax = fig.add_subplot(111)
        for i in range(minK):
            ax.plot(X, Y[:, i], color=cmap(i), label=f"mode_{i+1}")
        ax.set_xscale("log")
        ax.set_yscale("log")  # eigenvalues are positive; log-y is informative
        ax.set_xlabel("epoch (log)")
        ax.set_ylabel("eigenvalue (log)")
        title_which = "Forward/Activation" if which == "act" else "Backward/Delta"
        ax.set_title(f"{title_which} top-k eigenvalues — layer {ell}")
        sm = cm.ScalarMappable(cmap=cmap)
        sm.set_array(np.arange(1, minK+1))
        cbar = plt.colorbar(sm, ax=ax, pad=0.01)
        cbar.set_label("mode index")
        _save(fig, os.path.join(out_dir, f"eigvals_{which}_layer{ell}.png"))
