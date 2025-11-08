import argparse, os
from ffnnmetrics.plotting import (
    plot_training_mse,
    plot_kernel_scalars,
    plot_layer_shares,
    plot_rotation_per_layer,
    plot_alignment_per_layer,
    plot_transfer_inflow_topk_per_layer,
    plot_eigenvalues_over_epochs_per_layer,
)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, required=True, help="results/<run> directory or name")
    ap.add_argument("--outdir", type=str, default="plots")
    args = ap.parse_args()

    run_dir = args.run if os.path.isdir(args.run) else os.path.join("results", args.run)
    os.makedirs(args.outdir, exist_ok=True)

    # Basic
    plot_training_mse(run_dir, os.path.join(args.outdir, "mse.png"))
    plot_kernel_scalars(run_dir, os.path.join(args.outdir, "kernel_scalars.png"))
    plot_layer_shares(run_dir, os.path.join(args.outdir, "shares.png"))

    # New: rotations (log-log), activations & backprop
    plot_rotation_per_layer(run_dir, os.path.join(args.outdir, "rotation_act_loglog.png"), which="act", logx=True, logy=True)
    plot_rotation_per_layer(run_dir, os.path.join(args.outdir, "rotation_bp_loglog.png"), which="bp",  logx=True, logy=True)

    # New: alignment (log-x, linear-y)
    plot_alignment_per_layer(run_dir, os.path.join(args.outdir, "alignment_logx.png"), logx=True)

    # New: transfer inflow to top-k last-layer modes, per layer (one file per layer)
    out_transfer_dir = os.path.join(args.outdir, "transfer_inflow")
    os.makedirs(out_transfer_dir, exist_ok=True)
    plot_transfer_inflow_topk_per_layer(run_dir, out_transfer_dir)

    # New: top-k eigenvalues vs epoch (forward/activation), layer-separate
    out_eigs_act_dir = os.path.join(args.outdir, "eigs_act")
    os.makedirs(out_eigs_act_dir, exist_ok=True)
    plot_eigenvalues_over_epochs_per_layer(run_dir, out_eigs_act_dir, which="act")

    # New: top-k eigenvalues vs epoch (backward/delta), layer-separate
    out_eigs_bp_dir = os.path.join(args.outdir, "eigs_bp")
    os.makedirs(out_eigs_bp_dir, exist_ok=True)
    plot_eigenvalues_over_epochs_per_layer(run_dir, out_eigs_bp_dir, which="bp")
