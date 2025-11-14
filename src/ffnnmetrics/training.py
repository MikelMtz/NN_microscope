from __future__ import annotations
import os, csv, random
from dataclasses import dataclass
from typing import Tuple
import yaml
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import (
    parse_interaction_spec,
    StaircaseDataset,
    parse_tree_spec,
    ProductTreeDataset,
    MNISTDataset,
    RotMNISTDataset,
)


from .models import MLP
from .metrics import make_kernel_cache, compute_and_log_all_metrics

from dataclasses import dataclass
from typing import Tuple

@dataclass
class TrainConfig:
    run_name: str
    seed: int
    # data
    train_size: int; test_size: int; dim: int; spec: str; noise_std: float; shuffle_labels: bool
    dataset: str = "staircase"  # <— NEW: choose "staircase" or "product_tree"
    # model
    width: int = 256; depth: int = 4; activation: str = "relu"
    # training
    epochs: int = 10000; lr: float = 5e-3; optimizer: str = "gd"; momentum: float = 0.0; mode: str = "gd"
    batch_size: int | None = None
    # device
    device: str = "cuda"
    # kernel
    compute_kernel: bool = True; compute_kernel_every: int = 10; kernel_set: str = "train"
    max_kernel_points: int = 50000
    last_top_k: int = 15; lower_top_k: int = 15
    betas: Tuple[float, float, float] = (0.2, 0.2, 0.2)
    track_U: bool = True; save_transfers: bool = True




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
    cfg.setdefault("dataset", "staircase")  # <— default keeps old configs working
    return TrainConfig(**cfg)



def run_training(cfg: TrainConfig):
    device = get_device(cfg.device)
    set_seed(cfg.seed)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    results_dir = os.path.join("results", cfg.run_name)
    os.makedirs(results_dir, exist_ok=True)

    # --- DATASETS: choose by cfg.dataset ---
    if cfg.dataset.lower() == "product_tree":
        params = parse_tree_spec(cfg.spec)  # e.g. "tree(k=32, r=2, d=5, inputs=uniform)"
        k, r, depth = params["k"], params["r"], params["d"]
        input_mode = params["inputs"]
        assert cfg.dim >= k, f"dim={cfg.dim} < required k={k}"

        train_ds = ProductTreeDataset(cfg.train_size, cfg.dim, k=k, r=r, depth=depth,
                                    device=device, noise_std=cfg.noise_std, seed=cfg.seed,
                                    input_dist=input_mode)
        test_ds  = ProductTreeDataset(cfg.test_size,  cfg.dim, k=k, r=r, depth=depth,
                                    device=device, noise_std=0.0, seed=cfg.seed+1,
                                    input_dist=input_mode)

    elif cfg.dataset.lower() == "staircase":
        interactions = parse_interaction_spec(cfg.spec)
        need_dim = max([i for t in interactions for i in t]) + 1
        assert cfg.dim >= need_dim, f"dim={cfg.dim} < required {need_dim}"
        train_ds = StaircaseDataset(cfg.train_size, cfg.dim, interactions, device, cfg.noise_std, cfg.seed)
        test_ds  = StaircaseDataset(cfg.test_size,  cfg.dim, interactions, device, 0.0, cfg.seed + 1)

    elif cfg.dataset.lower() == "mnist":
        # MNIST images are 28x28 = 784 dimensions
        # If dim is not 784, it will be handled by the dataset (padding or truncation)
        # spec is not used for MNIST but required for interface compatibility
        train_ds = MNISTDataset(
            n=cfg.train_size,
            d=cfg.dim,
            spec=cfg.spec,  # Not used, but kept for interface
            device=device,
            noise_std=cfg.noise_std,
            seed=cfg.seed,
            train=True
        )
        test_ds = MNISTDataset(
            n=cfg.test_size,
            d=cfg.dim,
            spec=cfg.spec,  # Not used, but kept for interface
            device=device,
            noise_std=0.0,
            seed=cfg.seed + 1,
            train=False
        )

    elif cfg.dataset.lower() == "rotmnist":
        # Rotated MNIST - same as MNIST but with rotations
        # spec is not used for RotMNIST but required for interface compatibility
        train_ds = RotMNISTDataset(
            n=cfg.train_size,
            d=cfg.dim,
            spec=cfg.spec,  # Not used, but kept for interface
            device=device,
            noise_std=cfg.noise_std,
            seed=cfg.seed,
            train=True
        )
        test_ds = RotMNISTDataset(
            n=cfg.test_size,
            d=cfg.dim,
            spec=cfg.spec,  # Not used, but kept for interface
            device=device,
            noise_std=0.0,
            seed=cfg.seed + 1,
            train=False
        )

    else:
        raise ValueError(f"Unknown dataset '{cfg.dataset}'. Use 'staircase', 'product_tree', 'mnist', or 'rotmnist'.")





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
