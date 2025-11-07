


from __future__ import annotations
import os, csv, random
from dataclasses import dataclass
from typing import Tuple
import yaml
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import parse_interaction_spec, StaircaseDataset
from .models import MLP
from .metrics import make_kernel_cache, compute_and_log_all_metrics




@dataclass
class TrainConfig:
    run_name: str
    seed: int
    # data
    train_size: int; test_size: int; dim: int; spec: str; noise_std: float; shuffle_labels: bool
    # model
    width: int; depth: int; activation: str
    # training
    epochs: int; lr: float; optimizer: str; momentum: float; mode: str; batch_size: int
    # device
    device: str
    # kernel
    compute_kernel: bool; compute_kernel_every: int; kernel_set: str; max_kernel_points: int
    last_top_k: int; lower_top_k: int; betas: Tuple[float, float, float]; track_U: bool; save_transfers: bool


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(name: str) -> torch.device:
    return torch.device("cuda" if (name == "cuda" and torch.cuda.is_available()) else "cpu")


def evaluate_mse(model: MLP, loader: DataLoader) -> float:
    model.eval(); loss_fn = nn.MSELoss()
    tot = 0.0; n = 0
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb); loss = loss_fn(pred, yb)
            b = xb.shape[0]; tot += float(loss.item()) * b; n += b
    return tot / max(n, 1)


def load_config(path: str) -> TrainConfig:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return TrainConfig(**cfg)


def run_training(cfg: TrainConfig):
    device = get_device(cfg.device)
    set_seed(cfg.seed)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    results_dir = os.path.join("results", cfg.run_name)
    os.makedirs(results_dir, exist_ok=True)

    interactions = parse_interaction_spec(cfg.spec)
    need_dim = max([i for t in interactions for i in t]) + 1
    assert cfg.dim >= need_dim, f"dim={cfg.dim} < required {need_dim}"

    train_ds = StaircaseDataset(cfg.train_size, cfg.dim, interactions, device, cfg.noise_std, cfg.seed)
    test_ds  = StaircaseDataset(cfg.test_size,  cfg.dim, interactions, device, 0.0, cfg.seed + 1)

    if cfg.shuffle_labels:
        idx = torch.randperm(train_ds.y.shape[0], device=train_ds.y.device)
        train_ds.y = train_ds.y[idx]

    if cfg.mode == "gd":
        train_loader = DataLoader(train_ds, batch_size=cfg.train_size, shuffle=False)
    else:
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False)

    model = MLP(cfg.dim, cfg.width, cfg.depth, cfg.activation).to(device)
    model = torch.compile(model) if device.type == "cuda" else model

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr) if cfg.optimizer == "adam" else \
          torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)

    mse_csv = os.path.join(results_dir, "metrics.csv")
    with open(mse_csv, "w", newline="") as f: csv.writer(f).writerow(["epoch", "train_mse", "test_mse"])

    kernel_dir = os.path.join(results_dir, "kernel")
    kernel_cache = make_kernel_cache(kernel_dir, k=cfg.last_top_k)

    # epoch 0
    tr_mse0 = evaluate_mse(model, DataLoader(train_ds, batch_size=4096, shuffle=False))
    te_mse0 = evaluate_mse(model, test_loader)
    with open(mse_csv, "a", newline="") as f: csv.writer(f).writerow([0, tr_mse0, te_mse0])

    if cfg.compute_kernel:
        Xset, yset = (train_ds.x, train_ds.y) if cfg.kernel_set == "train" else (test_ds.x, test_ds.y)
        if cfg.max_kernel_points > 0 and Xset.shape[0] > cfg.max_kernel_points:
            idx = torch.randperm(Xset.shape[0], device=Xset.device)[:cfg.max_kernel_points]
            Xk, yk = Xset[idx], yset[idx]
        else:
            Xk, yk = Xset, yset
        from .metrics import compute_and_log_all_metrics
        compute_and_log_all_metrics(
            out_dir=kernel_dir, model=model, X=Xk, y=yk, top_k=cfg.last_top_k, lower_top_k=cfg.lower_top_k,
            epoch=0, betas=cfg.betas, track_U=cfg.track_U, save_transfers=cfg.save_transfers, kernel_cache=kernel_cache
        )

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = nn.MSELoss()(pred, yb)
            loss.backward(); opt.step()

        tr_mse = evaluate_mse(model, DataLoader(train_ds, batch_size=4096, shuffle=False))
        te_mse = evaluate_mse(model, test_loader)
        with open(mse_csv, "a", newline="") as f: csv.writer(f).writerow([epoch, tr_mse, te_mse])

        should_metrics = cfg.compute_kernel and (epoch % cfg.compute_kernel_every == 0 or epoch == cfg.epochs)
        if should_metrics:
            Xset, yset = (train_ds.x, train_ds.y) if cfg.kernel_set == "train" else (test_ds.x, test_ds.y)
            if cfg.max_kernel_points > 0 and Xset.shape[0] > cfg.max_kernel_points:
                idx = torch.randperm(Xset.shape[0], device=Xset.device)[:cfg.max_kernel_points]
                Xk, yk = Xset[idx], yset[idx]
            else:
                Xk, yk = Xset, yset
            compute_and_log_all_metrics(
                out_dir=kernel_dir, model=model, X=Xk, y=yk, top_k=cfg.last_top_k, lower_top_k=cfg.lower_top_k,
                epoch=epoch, betas=cfg.betas, track_U=cfg.track_U, save_transfers=cfg.save_transfers, kernel_cache=kernel_cache
            )

        if epoch % max(1, cfg.epochs // 20) == 0 or epoch == cfg.epochs:
            print(f"[Epoch {epoch}] train_mse={tr_mse:.6f}  test_mse={te_mse:.6f}")

    print(f"Saved to: {results_dir}")