import argparse, os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from ffnnmetrics.plotting import (
    # existing
    plot_training_mse,
    plot_training_accuracy,
    plot_kernel_scalars,
    plot_layer_shares,
    plot_rotation_per_layer,
    plot_alignment_per_layer,
    plot_transfer_inflow_topk_per_layer,
    plot_eigenvalues_over_epochs_per_layer,
    # new
    plot_lastlayer_advanced,
    plot_stb_histograms_latest,
    plot_interlayer_ilard_heatmaps_latest,
    plot_interlayer_depth_profiles,
    plot_interlayer_entropy_coherence_timeseries,
    plot_transport_alignment_heatmap_latest,
    plot_interlayer_heatmaps_over_time,
    plot_all_interlayer_metrics_grouped_timeseries,
)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, required=True, help="results_L30_epochs10000/<run> directory or name")
    ap.add_argument("--outdir", type=str, default="plots")
    args = ap.parse_args()

    run_dir = args.run if os.path.isdir(args.run) else os.path.join("results_L30_epochs10000", args.run)
    os.makedirs(args.outdir, exist_ok=True)

    # Basic
    plot_training_mse(run_dir, os.path.join(args.outdir, "mse.png"))
    plot_training_accuracy(run_dir, os.path.join(args.outdir, "accuracy.png"))
    plot_kernel_scalars(run_dir, os.path.join(args.outdir, "kernel_scalars.png"))
    plot_layer_shares(run_dir, os.path.join(args.outdir, "shares.png"))

    # Rotations (log-log), activations & backprop
    plot_rotation_per_layer(run_dir, os.path.join(args.outdir, "rotation_act_loglog.png"), which="act", logx=True, logy=True)
    plot_rotation_per_layer(run_dir, os.path.join(args.outdir, "rotation_bp_loglog.png"), which="bp",  logx=True, logy=True)

    # Alignment (log-x, linear-y)
    plot_alignment_per_layer(run_dir, os.path.join(args.outdir, "alignment_logx.png"), logx=True)

    # Transfer inflow to top-k last-layer modes, per layer
    out_transfer_dir = os.path.join(args.outdir, "transfer_inflow")
    os.makedirs(out_transfer_dir, exist_ok=True)
    plot_transfer_inflow_topk_per_layer(run_dir, out_transfer_dir)

    # Top-k eigenvalues vs epoch (forward/backward), layer-separate
    out_eigs_act_dir = os.path.join(args.outdir, "eigs_act")
    os.makedirs(out_eigs_act_dir, exist_ok=True)
    plot_eigenvalues_over_epochs_per_layer(run_dir, out_eigs_act_dir, which="act")

    out_eigs_bp_dir = os.path.join(args.outdir, "eigs_bp")
    os.makedirs(out_eigs_bp_dir, exist_ok=True)
    plot_eigenvalues_over_epochs_per_layer(run_dir, out_eigs_bp_dir, which="bp")

    # -------- NEW: last-layer advanced metrics (LARD, GSI, RPE, H_path, Top5, Gini, STR)
    out_adv = os.path.join(args.outdir, "lastlayer_advanced")
    os.makedirs(out_adv, exist_ok=True)
    plot_lastlayer_advanced(run_dir, out_adv)

    # -------- NEW: STB histograms at the latest epoch
    out_stb = os.path.join(args.outdir, "stb_hist")
    os.makedirs(out_stb, exist_ok=True)
    plot_stb_histograms_latest(run_dir, out_stb)

    # -------- NEW: inter-layer heatmaps (latest epoch) + time-series
    out_il = os.path.join(args.outdir, "interlayer")
    os.makedirs(out_il, exist_ok=True)
    plot_interlayer_ilard_heatmaps_latest(run_dir, out_il)  # ILARD, R_inflow, I_GSI, H_path_pair, Coherence_pair
    plot_interlayer_depth_profiles(run_dir, os.path.join(out_il, "depth_profiles.png"))
    plot_interlayer_entropy_coherence_timeseries(run_dir, os.path.join(out_il, "entropy_coherence_timeseries.png"))

    # -------- NEW: transported alignment heatmap (latest epoch)
    plot_transport_alignment_heatmap_latest(run_dir, os.path.join(out_il, "transported_alignment_heatmap.png"))
    #plot_interlayer_heatmaps_over_time(run_dir, os.path.join(out_il, "timelines"), n_mid=8)
    plot_all_interlayer_metrics_grouped_timeseries(run_dir, os.path.join(out_il, "grouped_timeseries"))
