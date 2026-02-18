
import argparse, os, yaml, torch
from ffnnmetrics.training import load_config
from ffnnmetrics.metrics import make_kernel_cache, compute_and_log_all_metrics
from ffnnmetrics.data import parse_interaction_spec, StaircaseDataset
from ffnnmetrics.models import MLP

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--epoch", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if (cfg.device == "cuda" and torch.cuda.is_available()) else "cpu")

    interactions = parse_interaction_spec(cfg.spec)
    train_ds = StaircaseDataset(cfg.train_size, cfg.dim, interactions, device, cfg.noise_std, cfg.seed)
    test_ds  = StaircaseDataset(cfg.test_size, cfg.dim, interactions, device, 0.0, cfg.seed + 1)
    Xset, yset = (train_ds.x, train_ds.y) if cfg.kernel_set == "train" else (test_ds.x, test_ds.y)

    model = MLP(cfg.dim, cfg.width, cfg.depth, cfg.activation).to(device)
    results_dir = os.path.join("results", cfg.run_name)
    kernel_dir = os.path.join(results_dir, "kernel")
    cache = make_kernel_cache(kernel_dir, k=cfg.last_top_k)

    if cfg.max_kernel_points > 0 and Xset.shape[0] > cfg.max_kernel_points:
        idx = torch.randperm(Xset.shape[0], device=Xset.device)[:cfg.max_kernel_points]
        Xset, yset = Xset[idx], yset[idx]

    compute_and_log_all_metrics(
        out_dir=kernel_dir, model=model, X=Xset, y=yset, top_k=cfg.last_top_k, lower_top_k=cfg.lower_top_k,
        epoch=args.epoch, betas=cfg.betas, track_U=cfg.track_U, save_transfers=cfg.save_transfers, kernel_cache=cache
    )
