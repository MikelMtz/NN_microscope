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

def _latest_epoch_from_npy(pattern: str) -> tuple[int | None, str | None]:
    """
    pattern example: ".../STB_layer*_epoch*.npy" or ".../T_path_diag_*_epoch*.npy"
    Returns (epoch_int, filepath) of the latest epoch found.
    """
    files = glob.glob(pattern)
    best_e, best_f = None, None
    for f in files:
        m = re.search(r"epoch(\d+)\.npy$", f)
        if not m: continue
        e = int(m.group(1))
        if best_e is None or e > best_e:
            best_e, best_f = e, f
    return best_e, best_f

# ---------------- basic plots (unchanged + small improvements) ----------------
def plot_training_accuracy(run_dir: str, out_path: str):
    p = os.path.join(run_dir, "metrics.csv")
    if not os.path.exists(p):
        print(f"[plot_training_accuracy] missing {p}")
        return
    df = pd.read_csv(p)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(df["epoch"], df["train_acc"], label="train")
    ax.plot(df["epoch"], df["test_acc"], label="test")
    ax.set_xlabel("epoch"); ax.set_ylabel("Accuracy"); ax.legend()
    _save(fig, out_path)

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
    # include new 'dom_ratio' if present
    for col in ["A_k", "d_eff", "DeltaC", "rho_k", "rho_bar", "S_k", "C", "G", "dom_ratio"]:
        if col in df.columns:
            ax.plot(df["epoch"], df[col], label=col)
    ax.set_xlabel("epoch"); ax.set_ylabel("value"); ax.legend(ncol=3, fontsize=8)
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
    ax.set_xlabel("epoch"); ax.set_ylabel("share"); ax.legend(ncol=2, fontsize=8)
    _save(fig, out_path)

# --------------- existing new plots ----------------

def plot_rotation_per_layer(run_dir: str, out_path: str, which: str = "act", logx: bool = True, logy: bool = True):
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
    ax.legend(ncol=2, fontsize=8)
    _save(fig, out_path)

def plot_alignment_per_layer(run_dir: str, out_path: str, logx: bool = True):
    p = os.path.join(run_dir, "kernel", "alignment.csv")
    p_transport = os.path.join(run_dir, "kernel", "transported_alignment.csv")
    
    # Try to read alignment.csv first
    if os.path.exists(p):
        df = pd.read_csv(p)
        # Check if we have data rows (not just headers)
        if len(df) > 0 and "epoch" in df.columns:
            x = df["epoch"].to_numpy()
            cols = [c for c in df.columns if c.startswith("align_layer_")]
            if cols:
                fig = plt.figure(); ax = fig.add_subplot(111)
                x_plot = x + 1 if logx else x
                for c in cols:
                    y = df[c].to_numpy()
                    ax.plot(x_plot, y, label=c)
                if logx: ax.set_xscale("log")
                ax.set_xlabel("epoch" + (" (log)" if logx else ""))
                ax.set_ylabel("alignment")
                ax.legend(ncol=2, fontsize=8)
                _save(fig, out_path)
                return
    
    # Fallback: generate from transported_alignment.csv
    if os.path.exists(p_transport):
        df_transport = pd.read_csv(p_transport)
        if df_transport.empty:
            print(f"[plot_alignment_per_layer] transported_alignment.csv is empty")
            return
        
        # Get number of layers from the data
        L = int(df_transport["r"].max())
        
        # Group by epoch and extract alignments to last layer (r == L)
        epochs = sorted(df_transport["epoch"].unique())
        align_data = {"epoch": epochs}
        
        for ell in range(1, L + 1):
            align_data[f"align_layer_{ell}"] = []
            for epoch in epochs:
                sub = df_transport[(df_transport["epoch"] == epoch) & 
                                   (df_transport["ell"] == ell) & 
                                   (df_transport["r"] == L)]
                if not sub.empty:
                    align_data[f"align_layer_{ell}"].append(float(sub.iloc[0]["A_transport"]))
                else:
                    align_data[f"align_layer_{ell}"].append(0.0)
        
        df = pd.DataFrame(align_data)
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
        ax.legend(ncol=2, fontsize=8)
        _save(fig, out_path)
        return
    
    print(f"[plot_alignment_per_layer] missing {p} and {p_transport}")
    return

def plot_transfer_inflow_topk_per_layer(run_dir: str, out_dir: str):
    epochs = _epochs_from_summary(run_dir)
    if epochs.size == 0:
        print("[plot_transfer_inflow_topk_per_layer] no summary.csv to infer epochs")
        return

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
    K = min(a.shape[1] for a in inflows)
    inflows = inflows[:, :, :K]
    L = inflows.shape[1]
    x = epochs_v + 1  # log-safe

    cmap = cm.get_cmap("viridis", K)
    for ell in range(1, L + 1):
        fig = plt.figure(); ax = fig.add_subplot(111)
        for i in range(K):
            y = inflows[:, ell-1, i]
            ax.plot(x, y, label=f"mode_{i+1}", color=cmap(i))
        ax.set_xscale("log")
        ax.set_xlabel("epoch (log)")
        ax.set_ylabel(r"$\sum_p T^{(\ell)}_{p\to i}$")
        ax.set_title(f"Layer {ell} transfer inflow to top-k modes")
        sm = cm.ScalarMappable(cmap=cmap)
        sm.set_array(np.arange(1, K+1))
        cbar = plt.colorbar(sm, ax=ax, pad=0.01)
        cbar.set_label("mode index")
        _save(fig, os.path.join(out_dir, f"transfer_inflow_layer{ell}.png"))

##########################################
#    Takes as argument:                  #
#                                        #
#    C_act_eigvals_layerX_epochYYYY.npy  #
##########################################
def plot_eigenvalues_over_epochs_per_layer(run_dir: str, out_dir: str, which: str = "act"):
    epochs = _epochs_from_summary(run_dir)
    if epochs.size == 0:
        print("[plot_eigenvalues_over_epochs_per_layer] no summary.csv to infer epochs")
        return

    layers = _available_layers_from_files(run_dir, which=which)
    if not layers:
        print(f"[plot_eigenvalues_over_epochs_per_layer] no eigenvalue files for which='{which}'")
        return

    for ell in layers:
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

        minK = min(v.shape[0] for v in vals_list)
        Y = np.stack([v[:minK] for v in vals_list], axis=0)  # [E, minK]
        X = np.array(valid_epochs, dtype=int) + 1  # log-safe

        cmap = cm.get_cmap("viridis", minK)
        fig = plt.figure(); ax = fig.add_subplot(111)
        for i in range(minK):
            ax.plot(X, Y[:, i], color=cmap(i), label=f"mode_{i+1}")
        ax.set_xscale("log"); ax.set_yscale("log")
        title_which = "Forward/Activation" if which == "act" else "Backward/Delta"
        ax.set_xlabel("epoch (log)"); ax.set_ylabel("eigenvalue (log)")
        ax.set_title(f"{title_which} top-k eigenvalues — layer {ell}")
        sm = cm.ScalarMappable(cmap=cmap)
        sm.set_array(np.arange(1, minK+1))
        cbar = plt.colorbar(sm, ax=ax, pad=0.01)
        cbar.set_label("mode index")
        _save(fig, os.path.join(out_dir, f"eigvals_{which}_layer{ell}.png"))

# ----------------------- NEW: last-layer advanced metrics -----------------------

def plot_lastlayer_advanced(run_dir: str, out_dir: str):
    p = os.path.join(run_dir, "kernel", "lastlayer_advanced.csv")
    if not os.path.exists(p):
        print(f"[plot_lastlayer_advanced] missing {p}")
        return
    df = pd.read_csv(p)
    x = df["epoch"].to_numpy()
    x_log = x + 1

    # 1) LARD
    if "LARD" in df.columns:
        fig = plt.figure(); ax = fig.add_subplot(111)
        ax.plot(x, df["LARD"], label="LARD")
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("epoch"); ax.set_ylabel("LARD (rotation/scale, label-aware)")
        ax.legend()
        _save(fig, os.path.join(out_dir, "lard.png"))

    # 2) GSI (log-y)
    if "GSI" in df.columns:
        fig = plt.figure(); ax = fig.add_subplot(111)
        y = np.maximum(df["GSI"].to_numpy(), 1e-16)
        ax.plot(x_log, y, label="GSI")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("epoch (log)"); ax.set_ylabel("gap-stress index (log)")
        ax.legend()
        _save(fig, os.path.join(out_dir, "gsi_loglog.png"))

    # 3) RPE (can be negative; plot baseline 0)
    if "RPE" in df.columns:
        fig = plt.figure(); ax = fig.add_subplot(111)
        ax.plot(x, df["RPE"], label="RPE")
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("epoch"); ax.set_ylabel("RPE = ΔA_k / R_y")
        ax.legend()
        _save(fig, os.path.join(out_dir, "rpe.png"))

    # 4) Pathway entropy / Top-5% / Gini
    cols_ok = [c for c in ["H_path","Top5_mass","Gini"] if c in df.columns]
    if cols_ok:
        fig = plt.figure(figsize=(8,4)); ax = fig.add_subplot(111)
        for c in cols_ok:
            ax.plot(x, df[c], label=c)
        ax.set_xlabel("epoch"); ax.set_ylabel("value")
        ax.legend()
        _save(fig, os.path.join(out_dir, "path_entropy_top5_gini.png"))

    # 5) STR
    if "STR" in df.columns:
        fig = plt.figure(); ax = fig.add_subplot(111)
        ax.plot(x, df["STR"], label="STR")
        ax.set_xlabel("epoch"); ax.set_ylabel("spectral turnover rate")
        ax.legend()
        _save(fig, os.path.join(out_dir, "str.png"))

# ----------------------- NEW: STB histograms (latest epoch) -----------------------

def plot_stb_histograms_latest(run_dir: str, out_dir: str):
    """
    For latest epoch available, draw histograms of STB_i^{(ℓ)} over top-k modes i for each layer ℓ.
    Expects files: kernel/STB_layer{ell}_epochXXXX.npy
    """
    # discover latest epoch by scanning any layer's file
    pattern = os.path.join(run_dir, "kernel", "STB_layer*_epoch*.npy")
    latest_e, _ = _latest_epoch_from_npy(pattern)
    if latest_e is None:
        print("[plot_stb_histograms_latest] no STB files found")
        return

    # how many layers?
    layer_files = glob.glob(pattern)
    layers = sorted({int(re.search(r"STB_layer(\d+)_epoch", os.path.basename(f)).group(1)) for f in layer_files})

    for ell in layers:
        p = os.path.join(run_dir, "kernel", f"STB_layer{ell}_epoch{latest_e:04d}.npy")
        if not os.path.exists(p):
            continue
        arr = np.load(p)  # shape [kL], values in [-1,1]
        fig = plt.figure(); ax = fig.add_subplot(111)
        ax.hist(arr, bins=21, range=(-1,1), alpha=0.85)
        ax.set_xlabel(f"STB_i (layer {ell})"); ax.set_ylabel("count")
        ax.set_title(f"STB distribution — layer {ell} (epoch {latest_e})")
        _save(fig, os.path.join(out_dir, f"stb_layer{ell}_epoch{latest_e:04d}.png"))

# ----------------------- NEW: inter-layer visuals -----------------------

def _load_interlayer_ilard_df(run_dir: str) -> pd.DataFrame | None:
    p = os.path.join(run_dir, "kernel", "interlayer_ilard.csv")
    if not os.path.exists(p):
        print("[_load_interlayer_ilard_df] missing interlayer_ilard.csv")
        return None
    return pd.read_csv(p)

def _latest_epoch_in_df(df: pd.DataFrame) -> int:
    return int(df["epoch"].max())

def _assemble_heatmap(df: pd.DataFrame, metric: str, L: int, epoch: int) -> np.ndarray:
    """
    Build an LxL matrix with value only for ell<=r; NaN elsewhere.
    """
    M = np.full((L, L), np.nan, dtype=float)
    sub = df[df["epoch"] == epoch]
    for _, row in sub.iterrows():
        ell, r = int(row["ell"]), int(row["r"])
        if metric in row and 1 <= ell <= L and 1 <= r <= L:
            M[ell-1, r-1] = float(row[metric])
    return M

def _heatmap_plot(mat: np.ndarray, title: str, out_path: str, vmin=None, vmax=None, cmap="viridis"):
    fig = plt.figure(figsize=(6,5)); ax = fig.add_subplot(111)
    im = ax.imshow(mat, origin="lower", cmap=cmap, aspect="equal", vmin=vmin, vmax=vmax)
    ax.set_xlabel("target layer r"); ax.set_ylabel("source layer ℓ")
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax, pad=0.01)
    _save(fig, out_path)

def plot_interlayer_ilard_heatmaps_latest(run_dir: str, out_dir: str):
    """
    Heatmaps at the latest epoch for:
      - ILARD  (inter-layer label-aware rotation dominance)
      - R_inflow (Σ |B_off| / gap)
      - I_GSI    (Σ |B_off| / gap^2)
      - H_path_pair (entropy)
      - Coherence_pair
    """
    df = _load_interlayer_ilard_df(run_dir)
    if df is None or df.empty:
        return

    L = _num_layers_from_rot_csv(run_dir)
    if L is None:  # fallback: infer max r from csv
        L = int(max(df["r"].max(), df["ell"].max()))

    epoch = _latest_epoch_in_df(df)
    metrics = [
        ("ILARD", "ILARD (latest)"),
        ("R_inflow", "Rotation inflow R (latest)"),
        ("I_GSI", "Gap-stress I-GSI (latest)"),
        ("H_path_pair", "Inter-layer pathway entropy (latest)"),
        ("Coherence_pair", "Inter-layer coherence (latest)"),
    ]
    for mcol, title in metrics:
        if mcol not in df.columns: continue
        mat = _assemble_heatmap(df, mcol, L, epoch)
        _heatmap_plot(mat, title, os.path.join(out_dir, f"{mcol.lower()}_heatmap_epoch{epoch:04d}.png"))

def plot_interlayer_depth_profiles(run_dir: str, out_path: str):
    p = os.path.join(run_dir, "kernel", "interlayer_depth_profiles.csv")
    if not os.path.exists(p):
        print(f"[plot_interlayer_depth_profiles] missing {p}")
        return
    df = pd.read_csv(p)
    fig = plt.figure(); ax = fig.add_subplot(111)
    for col in ["D_scale", "D_rot"]:
        if col in df.columns:
            ax.plot(df["epoch"], df[col], label=col)
    ax.set_xlabel("epoch"); ax.set_ylabel("depth")
    ax.legend()
    _save(fig, out_path)

def plot_interlayer_entropy_coherence_timeseries(run_dir: str, out_path: str):
    """
    Average H_path_pair and Coherence_pair across all (ell->r) pairs per epoch.
    """
    df = _load_interlayer_ilard_df(run_dir)
    if df is None or df.empty:
        return
    cols = [c for c in ["H_path_pair", "Coherence_pair"] if c in df.columns]
    if not cols:
        print("[plot_interlayer_entropy_coherence_timeseries] columns not found")
        return
    grouped = df.groupby("epoch")[cols].mean().reset_index()
    fig = plt.figure(); ax = fig.add_subplot(111)
    for c in cols:
        ax.plot(grouped["epoch"], grouped[c], label=f"mean {c}")
    ax.set_xlabel("epoch"); ax.set_ylabel("value")

    ax.legend()
    _save(fig, out_path)

def plot_transport_alignment_heatmap_latest(run_dir: str, out_path: str):
    p = os.path.join(run_dir, "kernel", "transported_alignment.csv")
    if not os.path.exists(p):
        print(f"[plot_transport_alignment_heatmap_latest] missing {p}")
        return
    df = pd.read_csv(p)
    if df.empty:
        return
    L = _num_layers_from_rot_csv(run_dir)
    if L is None:
        L = int(max(df["r"].max(), df["ell"].max()))
    epoch = int(df["epoch"].max())
    M = np.full((L, L), np.nan, dtype=float)
    sub = df[df["epoch"] == epoch]
    for _, row in sub.iterrows():
        ell, r, val = int(row["ell"]), int(row["r"]), float(row["A_transport"])
        if 1 <= ell <= L and 1 <= r <= L:
            M[ell-1, r-1] = val
    _heatmap_plot(M, f"Transported alignment A(ℓ→r) — epoch {epoch}", out_path)


def _load_interlayer_ilard_df(run_dir: str) -> pd.DataFrame | None:
    p = os.path.join(run_dir, "kernel", "interlayer_ilard.csv")
    if not os.path.exists(p):
        print("[_load_interlayer_ilard_df] missing interlayer_ilard.csv")
        return None
    return pd.read_csv(p)

def _blend_with_white(rgb, t: float):
    """Return color blended toward white by factor t in [0,1]."""
    r, g, b = rgb
    return (r + (1 - r) * t, g + (1 - g) * t, b + (1 - b) * t)

def _base_hues_for_groups(n_groups: int):
    """
    Pick distinct base hues for groups (source layers). We start from tab10
    and then cycle with small HSV rotations if needed.
    """
    import matplotlib.colors as mcolors
    tab = plt.get_cmap("tab10")
    bases = [tab(i % 10)[:3] for i in range(n_groups)]
    if n_groups <= 10:
        return bases
    # lightly rotate hue for extras
    hsv = [mcolors.rgb_to_hsv(b) for b in bases]
    out = []
    for i in range(n_groups):
        h, s, v = hsv[i % len(hsv)]
        h = (h + 0.07 * (i // len(hsv))) % 1.0
        out.append(mcolors.hsv_to_rgb((h, s, v)))
    return out[:n_groups]

def plot_interlayer_metric_timeseries_grouped(
    run_dir: str,
    out_path: str,
    metric: str = "ILARD",
    min_points: int = 2,
    legend: bool = True,
):
    """
    Plot an inter-layer metric (from kernel/interlayer_ilard.csv) versus epoch.
    Group by source layer ℓ: each ℓ (group) uses a base hue; within that group,
    all targets r≥ℓ use progressively lighter shades of that hue.

    metric: one of ["ILARD", "R_inflow", "I_GSI", "H_path_pair", "Coherence_pair"].
    """
    df = _load_interlayer_ilard_df(run_dir)
    if df is None or df.empty:
        print(f"[plot_interlayer_metric_timeseries_grouped] no data for {metric}")
        return
    if metric not in df.columns:
        print(f"[plot_interlayer_metric_timeseries_grouped] metric '{metric}' not in CSV")
        return

    # infer L (num hidden layers)
    L = _num_layers_from_rot_csv(run_dir)
    if L is None:
        L = int(max(df["r"].max(), df["ell"].max()))

    # available epochs
    epochs_all = sorted(df["epoch"].unique().tolist())
    if len(epochs_all) < min_points:
        print("[plot_interlayer_metric_timeseries_grouped] too few epochs")
        return

    # precompute group targets count per ℓ
    group_targets = {ell: [r for r in range(ell, L + 1)] for ell in range(1, L + 1)}
    base_colors = _base_hues_for_groups(L)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    # draw each (ell->r) curve; shades by r index inside group
    handles = []
    labels = []

    metrics_loglog = {"ILARD", "R_inflow", "I_GSI"}
    use_log = metric in metrics_loglog

    for ell in range(1, L + 1):
        targets = group_targets[ell]
        n_shades = max(1, len(targets))
        # shade factors: 0.15..0.85 (lighter as r increases)
        tvals = np.linspace(0.15, 0.85, n_shades)
        base = base_colors[ell - 1]
        for j, r in enumerate(targets):
            color = _blend_with_white(base, tvals[j])
            sub = df[(df["ell"] == ell) & (df["r"] == r)].sort_values("epoch")
            if sub.empty:
                continue
            x = sub["epoch"].to_numpy()
            if use_log:
                x_plot = x + 1
            else:
                x_plot = x
            y = sub[metric].to_numpy()
            if use_log:
                y = np.clip(y, 1e-12, None)
            if y.size < min_points:
                continue
            h, = ax.plot(x_plot, y, color=color, linewidth=1.6)
            handles.append(h); labels.append(f"ℓ={ell}→r={r}")

    if use_log:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("epoch (log)")
        ax.set_ylabel(f"{metric} (log)")
    else:
        ax.set_xlabel("epoch")
        ax.set_ylabel(metric)
    ax.set_title(f"Inter-layer {metric} vs epoch (grouped by source layer ℓ)")
    if legend and handles:
        ax.legend(handles, labels, ncol=3, fontsize=8, frameon=False)

    _save(fig, out_path)

def plot_all_interlayer_metrics_grouped_timeseries(run_dir: str, out_dir: str):
    """
    Convenience wrapper: generate grouped time series for all inter-layer metrics.
    """
    os.makedirs(out_dir, exist_ok=True)
    metrics = ["ILARD", "R_inflow", "I_GSI", "H_path_pair", "Coherence_pair"]
    for m in metrics:
        plot_interlayer_metric_timeseries_grouped(
            run_dir,
            os.path.join(out_dir, f"{m.lower()}_grouped_timeseries.png"),
            metric=m
        )

def plot_interlayer_heatmaps_over_time(
    run_dir: str,
    out_dir: str,
    n_mid: int = 8,
    metrics: list[str] | None = None,
):
    """
    For each inter-layer metric, draw a row of heatmaps at:
      [ earliest, n_mid evenly spaced mid-epochs, latest ].

    Reads: kernel/interlayer_ilard.csv
    Metrics available in that CSV:
      - "ILARD", "R_inflow", "I_GSI", "H_path_pair", "Coherence_pair"

    Saves one PNG per metric into out_dir.
    """
    df = _load_interlayer_ilard_df(run_dir)
    if df is None or df.empty:
        return

    # infer number of layers (L)
    L = _num_layers_from_rot_csv(run_dir)
    if L is None:
        L = int(max(df["r"].max(), df["ell"].max()))

    # choose epochs: first, n_mid midpoints, last
    epochs_all = sorted(df["epoch"].unique().tolist())
    if len(epochs_all) == 0:
        print("[plot_interlayer_heatmaps_over_time] no epochs found")
        return

    if len(epochs_all) <= (n_mid + 2):
        chosen = epochs_all
    else:
        idx = np.linspace(0, len(epochs_all) - 1, num=n_mid + 2)
        idx = np.unique(np.round(idx).astype(int))
        chosen = [epochs_all[i] for i in idx]

    # which metrics to plot
    all_mets = ["ILARD", "R_inflow", "I_GSI", "H_path_pair", "Coherence_pair"]
    metrics = [m for m in (metrics or all_mets) if m in df.columns]
    if not metrics:
        print("[plot_interlayer_heatmaps_over_time] none of the metrics available")
        return

    os.makedirs(out_dir, exist_ok=True)

    for mcol in metrics:
        # assemble matrices for all chosen epochs
        mats = []
        for ep in chosen:
            mat = _assemble_heatmap(df, mcol, L, ep)  # NaNs above diagonal / missing pairs
            mats.append(mat)

        # consistent color scale across panels
        if mcol == "Coherence_pair":
            vmin, vmax = 0.0, 1.0
        else:
            stack_vals = np.concatenate([np.nan_to_num(M, nan=np.nan).ravel() for M in mats])
            stack_vals = stack_vals[np.isfinite(stack_vals)]
            if stack_vals.size == 0:
                vmin, vmax = 0.0, 1.0
            else:
                vmin = 0.0
                vmax = float(np.nanmax(stack_vals))
                if vmax <= 0: vmax = 1.0

        # figure layout: one row, len(chosen) columns
        n = len(chosen)
        fig_w = max(3.0 * n, 6.0)  # width per panel ~= 3
        fig = plt.figure(figsize=(fig_w, 3.4))
        axes = []

        for j, (ep, M) in enumerate(zip(chosen, mats)):
            ax = fig.add_subplot(1, n, j + 1)
            im = ax.imshow(M, origin="lower", cmap="viridis", aspect="equal", vmin=vmin, vmax=vmax)
            if j == 0:
                ax.set_ylabel("source layer ℓ")
            else:
                ax.set_yticks([])
            ax.set_xlabel(f"r (epoch {ep})")
            ax.set_title("" if j not in (0, n - 1) else (mcol if j == 0 else ""))
            axes.append(ax)

        # one colorbar for all
        cbar = fig.colorbar(im, ax=axes, pad=0.01, fraction=0.02)
        cbar.set_label(mcol)
        _save(fig, os.path.join(out_dir, f"{mcol.lower()}_timeline.png"))
