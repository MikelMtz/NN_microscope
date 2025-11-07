import argparse, os
from ffnnmetrics.plotting import plot_training_mse, plot_kernel_scalars, plot_layer_shares


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, required=True, help="results/<run> directory or name")
    ap.add_argument("--outdir", type=str, default="plots")
    args = ap.parse_args()


    run_dir = args.run if os.path.isdir(args.run) else os.path.join("results", args.run)
    os.makedirs(args.outdir, exist_ok=True)


    plot_training_mse(run_dir, os.path.join(args.outdir, "mse.png"))
    plot_kernel_scalars(run_dir, os.path.join(args.outdir, "kernel_scalars.png"))
    plot_layer_shares(run_dir, os.path.join(args.outdir, "shares.png"))