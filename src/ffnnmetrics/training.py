from __future__ import annotations
import os, csv, random
from dataclasses import dataclass
from typing import Tuple
import yaml
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

# Import data for MNIST
from .data import (
    parse_interaction_spec,
    StaircaseDataset,
    parse_tree_spec,
    ProductTreeDataset,
    MNISTDataset,
    RotMNISTDataset,
    CIFAR10Dataset,
    SVHNDataset
)


from .models import MLP, ResNet18
from .metrics import make_kernel_cache, compute_and_log_all_metrics

from dataclasses import dataclass
from typing import Tuple

@dataclass
class TrainConfig:
    run_name: str
    seed: int
    # data
    data_dir: str
    spec: str
    noise_std: float
    shuffle_labels: bool
    train_size: int; test_size: int; dim: int; spec: str; noise_std: float; shuffle_labels: bool
    
    report_accuracy: bool
    weight_decay: float
    loss: str = "mse"

    dataset: str = "staircase"  # <— NEW: choose "staircase" or "product_tree"
    # model
    model: str = "MLP"  # <— NEW: choose "MLP" or "ResNet18"
    width: int = 256; depth: int = 4; activation: str = "relu"
    num_classes: int = 10  # <— NEW: for ResNet18 output classes
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

def evaluate_loss(model:MLP, loader: DataLoader, loss_type: str) -> float:
    model.eval()
    if loss_type.lower() == "mse":
        loss_fn = nn.MSELoss()
    elif loss_type.lower() == "cross_entropy":
        loss_fn = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss function: {loss_type}")

    tot = 0.0
    n = 0
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb)

            # Debugging predictions and targets
            print(f"[DEBUG] Predictions shape: {pred.shape}")
            print(f"[DEBUG] Targets shape: {yb.shape}")

            # Reshape targets for CrossEntropyLoss
            if loss_type.lower() == "cross_entropy":
                yb = yb.view(-1).long()  # Convert to 1D tensor of class indices
                # Debugging reshaped targets
                print(f"[DEBUG] Reshaped Targets shape: {yb.shape}")

            loss = loss_fn(pred, yb)
            b = xb.shape[0]
            tot += float(loss.item()) * b
            n += b

    return tot / max(n, 1)

def load_config(path: str) -> TrainConfig:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("dataset", "staircase")  # <— default keeps old configs working
    return TrainConfig(**cfg)

# New function to get model based on config (NEW)
def get_model(cfg: TrainConfig) -> nn.Module:
    if cfg.model == "MLP":
        return MLP(cfg.dim, cfg.width, cfg.depth, cfg.activation, num_classes=cfg.num_classes).to(cfg.device)
    elif cfg.model == "ResNet18":
        return ResNet18(num_classes=cfg.num_classes).to(cfg.device)
    else:
        raise ValueError(f"Unsupported model: {cfg.model}")

# New function to evaluate accuracy (NEW)
def evaluate_accuracy(model: nn.Module, loader: DataLoader) -> float:
    """
    Evaluate the accuracy of the model on the given data loader.

    Args:
        model (nn.Module): The model to evaluate.
        loader (DataLoader): The data loader containing the dataset.

    Returns:
        float: The accuracy of the model as a percentage.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb)

            # For classification tasks, get the predicted class
            if pred.dim() > 1:
                pred_classes = pred.argmax(dim=1)
            else:
                pred_classes = (pred > 0.5).long()  # For binary classification

            # Ensure targets are in the correct shape
            if yb.dim() > 1:
                yb = yb.argmax(dim=1)

            correct += (pred_classes == yb).sum().item()
            total += yb.size(0)

    return 100.0 * correct / total if total > 0 else 0.0

def run_training(cfg: TrainConfig):
    device = get_device(cfg.device)
    set_seed(cfg.seed)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    results_dir = os.path.join("results", cfg.run_name)
    os.makedirs(results_dir, exist_ok=True)

    # --- DATASETS: choose by cfg.dataset ---
    if cfg.dataset.lower() == "cifar10":
        train_ds = CIFAR10Dataset(
            n=cfg.train_size,
            d=cfg.dim,
            spec=cfg.spec,  # Not used, but kept for interface
            device=device,
            noise_std=cfg.noise_std,
            seed=cfg.seed,
            train=True
        )
        test_ds = CIFAR10Dataset(
            n=cfg.test_size,
            d=cfg.dim,
            spec=cfg.spec,  # Not used, but kept for interface
            device=device,
            noise_std=0.0,
            seed=cfg.seed + 1,
            train=False
        )

    elif cfg.dataset.lower() == "svhn":
        train_ds = SVHNDataset(
            n=cfg.train_size,
            d=cfg.dim,
            spec=cfg.spec,  # Not used, but kept for interface
            device=device,
            noise_std=cfg.noise_std,
            seed=cfg.seed,
            train=True
        )
        test_ds = SVHNDataset(
            n=cfg.test_size,
            d=cfg.dim,
            spec=cfg.spec,  # Not used, but kept for interface
            device=device,
            noise_std=0.0,
            seed=cfg.seed + 1,
            train=False
        )

    elif cfg.dataset.lower() == "product_tree":
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
        raise ValueError(f"Unknown dataset '{cfg.dataset}'. Use 'staircase', 'product_tree', 'mnist', 'cifar10', 'svhn' or 'rotmnist'.")


    # --- CREATE train_loader and test_loader
    if cfg.shuffle_labels:
        idx = torch.randperm(train_ds.y.shape[0], device=train_ds.y.device)
        train_ds.y = train_ds.y[idx]

    if cfg.mode == "gd":
        train_loader = DataLoader(train_ds, batch_size=cfg.train_size, shuffle=False)
    else:
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False)

    # --- MODEL ---
    model = get_model(cfg)
    model = torch.compile(model) if device.type == "cuda" else model

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr) if cfg.optimizer == "adam" else \
          torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
    
    
    # Add cosine annealing scheduler (NEW)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)



    mse_csv = os.path.join(results_dir, "metrics.csv")
    with open(mse_csv, "w", newline="") as f: csv.writer(f).writerow(["epoch", "train_mse", "test_mse"])

    kernel_dir = os.path.join(results_dir, "kernel")
    kernel_cache = make_kernel_cache(kernel_dir, k=cfg.last_top_k)

    # epoch 0
    tr_loss0 = evaluate_loss(model, DataLoader(train_ds, batch_size=4096, shuffle=False), cfg.loss)
    te_loss0 = evaluate_loss(model, test_loader, cfg.loss)
    with open(mse_csv, "a", newline="") as f: csv.writer(f).writerow([0, tr_loss0, te_loss0])

    # Evaluate accuracy at epoch 0 if reporting accuracy (NEW)
    tr_acc0 = evaluate_accuracy(model, DataLoader(train_ds, batch_size=4096, shuffle=False))
    te_acc0 = evaluate_accuracy(model, test_loader)
    with open(mse_csv, "a", newline="") as f:
        csv.writer(f).writerow([0, tr_loss0, te_loss0, tr_acc0, te_acc0])

    # Update CSV header to include accuracy
    with open(mse_csv, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_mse", "test_mse", "train_acc", "test_acc"])

    ##############################################################
    # Commented out the metrics calculation to focus only on MSE #
    ##############################################################
    # if cfg.compute_kernel:
    #     Xset, yset = (train_ds.x, train_ds.y) if cfg.kernel_set == "train" else (test_ds.x, test_ds.y)
    #     if cfg.max_kernel_points > 0 and Xset.shape[0] > cfg.max_kernel_points:
    #         idx = torch.randperm(Xset.shape[0], device=Xset.device)[:cfg.max_kernel_points]
    #         Xk, yk = Xset[idx], yset[idx]
    #     else:
    #         Xk, yk = Xset, yset
    #     compute_and_log_all_metrics(
    #         out_dir=kernel_dir, model=model, X=Xk, y=yk, top_k=cfg.last_top_k, lower_top_k=cfg.lower_top_k,
    #         epoch=0, betas=cfg.betas, track_U=cfg.track_U, save_transfers=cfg.save_transfers, kernel_cache=kernel_cache
    #     )

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        
        # Initialize the loss function based on cfg.loss
        if cfg.loss.lower() == "mse":
            loss_fn = nn.MSELoss()
        elif cfg.loss.lower() == "cross_entropy":
            loss_fn = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss function: {cfg.loss}")

        for xb, yb in train_loader:
            opt.zero_grad(set_to_none=True)
            pred = model(xb)

            # Clamp predictions to prevent inf/nan values
            pred = torch.clamp(pred, min=-1e6, max=1e6)

            # Adjust learning rate if necessary
            if epoch == 1 and cfg.lr > 0.01:
                for g in opt.param_groups:
                    g['lr'] = cfg.lr / 10

            # Ensure compatibility for CrossEntropyLoss
            if cfg.loss.lower() == "cross_entropy":
                if pred.size(1) == 1:
                    raise ValueError("Model output shape is [N, 1]. For CrossEntropyLoss, it should be [N, C] with C >= 2.")
                if yb.size(1) == 1:
                    yb = yb.view(-1).long()  # Convert targets to class indices

            # Ensure compatibility for MSELoss
            #elif cfg.loss.lower() == "mse":
            #    if yb.size(1) != pred.size(1):
            #        raise ValueError("For MSELoss, target shape must match prediction shape.")

            # Validate loss function application
            if cfg.loss.lower() == "cross_entropy":
                if pred.dim() == 2 and pred.size(1) == 1:
                    loss_fn = torch.nn.BCEWithLogitsLoss()
                    loss = loss_fn(pred.view(-1), yb.view(-1).float())
                elif yb.dim() == 2 and yb.size(1) == pred.size(1):
                    if yb.dtype.is_floating_point and (yb.sum(dim=1) - 1.0).abs().max() > 1e-4:
                        logp = torch.nn.functional.log_softmax(pred, dim=1)
                        loss = -(yb * logp).sum(dim=1).mean()
                    else:
                        targets = yb.argmax(dim=1).long()
                        loss = loss_fn(pred, targets)
                else:
                    targets = yb.view(-1).long()
                    loss = loss_fn(pred, targets)
            else:
                if isinstance(loss_fn, torch.nn.BCEWithLogitsLoss):
                    loss = loss_fn(pred.view(-1), yb.view(-1).float())
                else:
                    loss = loss_fn(pred, yb)

            # Backward pass
            loss.backward()

            # Gradient clipping (NEW)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimizer step
            opt.step()

            ##### Debugging: Check for NaN or Inf in loss and gradients
            # Compute and print gradient norms for debugging
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)  # L2 norm
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            print(f"[DEBUG] Gradient Norm: {total_norm}")
        
        # Update learning rate using the scheduler (NEW)
        scheduler.step()

        tr_loss = evaluate_loss(model, DataLoader(train_ds, batch_size=4096, shuffle=False), cfg.loss)
        te_loss = evaluate_loss(model, test_loader, cfg.loss)
        tr_acc = evaluate_accuracy(model, DataLoader(train_ds, batch_size=4096, shuffle=False))
        te_acc = evaluate_accuracy(model, test_loader)

        # Debugging values before writing to metrics.csv
        print(f"[DEBUG] Writing to CSV - Epoch: {epoch}, Train MSE: {tr_loss}, Test MSE: {te_loss}, Train Acc: {tr_acc}, Test Acc: {te_acc}")

        # Write metrics to CSV
        with open(mse_csv, "a", newline="") as f:
            csv.writer(f).writerow([epoch, tr_loss, te_loss, tr_acc, te_acc])

        ##############################################################
        # Commented out the metrics calculation to focus only on MSE #
        ##############################################################

        # should_metrics = cfg.compute_kernel and (epoch % cfg.compute_kernel_every == 0 or epoch == cfg.epochs)
        # if should_metrics:
        #     Xset, yset = (train_ds.x, train_ds.y) if cfg.kernel_set == "train" else (test_ds.x, test_ds.y)
        #     if cfg.max_kernel_points > 0 and Xset.shape[0] > cfg.max_kernel_points:
        #         idx = torch.randperm(Xset.shape[0], device=Xset.device)[:cfg.max_kernel_points]
        #         Xk, yk = Xset[idx], yset[idx]
        #     else:
        #         Xk, yk = Xset, yset
        #     compute_and_log_all_metrics(
        #         out_dir=kernel_dir, model=model, X=Xk, y=yk, top_k=cfg.last_top_k, lower_top_k=cfg.lower_top_k,
        #         epoch=epoch, betas=cfg.betas, track_U=cfg.track_U, save_transfers=cfg.save_transfers, kernel_cache=kernel_cache
        #     )

        if epoch % max(1, cfg.epochs // 20) == 0 or epoch == cfg.epochs:
            print(f"[Epoch {epoch}] train_mse={tr_loss:.6f}  test_mse={te_loss:.6f}")

    print(f"Saved to: {results_dir}")