from __future__ import annotations
import os
from typing import List, Tuple
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms.functional import rotate

__all__ = ["parse_interaction_spec", "generate_inputs", "compute_labels", "StaircaseDataset",
   "parse_tree_spec",
    "generate_inputs_continuous",
    "compute_product_tree_labels",
    "ProductTreeDataset",
    "MNISTDataset",
    "RotMNISTDataset",
]

def parse_interaction_spec(spec: str) -> List[List[int]]:
    cleaned = spec.replace(" ", "")
    parts = cleaned.split("+")
    terms: List[List[int]] = []
    for p in parts:
        assert p.startswith("{") and p.endswith("}"), f"Malformed term '{p}'."
        inside = p[1:-1]
        assert inside != "", f"Empty term '{p}'."
        idxs = inside.split(",")
        term = []
        for s in idxs:
            assert s.isdigit(), f"Non-integer index '{s}'."
            j = int(s); assert j > 0
            term.append(j - 1)
        terms.append(term)
    return terms

@torch.no_grad()
def generate_inputs(n: int, d: int, device: torch.device) -> torch.Tensor:
    x = torch.randint(0, 2, (n, d), device=device, dtype=torch.int8).to(torch.float32)
    return 2.0 * x - 1.0

@torch.no_grad()
def compute_labels(x: torch.Tensor, interactions: List[List[int]]) -> torch.Tensor:
    y = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    for term in interactions:
        y += torch.prod(x[:, term], dim=1)
    return y.unsqueeze(1)


# --- NEW: parser for strings like "tree(k=32, r=2, d=5, inputs=uniform)" ---
def parse_tree_spec(spec: str) -> dict:
    s = spec.strip().lower().replace(" ", "")
    assert s.startswith("tree(") and s.endswith(")"), f"Malformed tree spec: '{spec}'"
    inside = s[5:-1]
    items = {}
    if inside:
        for part in inside.split(","):
            key, val = part.split("=")
            items[key] = val
    r = int(items.get("r", 2))
    inputs = items.get("inputs", "uniform")
    k = int(items["k"]) if "k" in items else None
    d = int(items["d"]) if "d" in items else None

    import math
    if k is None and d is None:
        raise AssertionError("tree spec must define at least one of k or d")
    if d is None:
        d = int(math.ceil(math.log(max(1, k), r)))
    if k is None:
        k = r ** d
    if r ** d < k:
        raise AssertionError(f"Inconsistent tree spec: r**d={r**d} < k={k}")
    return {"k": k, "r": r, "d": d, "inputs": inputs}

@torch.no_grad()
def generate_inputs_continuous(n: int, d: int, device: torch.device, seed: int | None = None,
                               dist: str = "uniform") -> torch.Tensor:
    if dist == "binary":
        return generate_inputs(n, d, device)
    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(seed)
    return 2.0 * torch.rand((n, d), generator=g, device=device, dtype=torch.float32) - 1.0

def _product_tree_reduce(Z: torch.Tensor, r: int) -> torch.Tensor:
    assert Z.dim() == 2
    x = Z
    one = torch.ones((x.shape[0], 1), device=x.device, dtype=x.dtype)
    while x.shape[1] > 1:
        rem = x.shape[1] % r
        if rem != 0:
            x = torch.cat([x, one.expand(-1, r - rem)], dim=1)
        x = x.view(x.shape[0], -1, r).prod(dim=2)
    return x[:, 0]

@torch.no_grad()
def compute_product_tree_labels(x: torch.Tensor, k: int, r: int = 2, d: int | None = None,
                                indices: List[int] | None = None) -> torch.Tensor:
    n, D = x.shape
    if indices is None:
        assert k <= D, f"Requested k={k} > dim={D}"
        idx = torch.arange(k, device=x.device)
    else:
        idx = torch.as_tensor(indices, device=x.device)
        assert idx.numel() == k and int(idx.max().item()) < D and int(idx.min().item()) >= 0
    Z = x.index_select(dim=1, index=idx)
    root = _product_tree_reduce(Z, r=r)
    return root.unsqueeze(1)

class StaircaseDataset(Dataset):
    def __init__(self, n: int, d: int, interactions: List[List[int]], device: torch.device, noise_std: float = 0.0, seed: int = 0):
        _ = seed
        self.x = generate_inputs(n, d, device)
        self.y = compute_labels(self.x, interactions)
        if noise_std > 0:
            self.y = self.y + noise_std * torch.randn_like(self.y)
        self.n = n
    def __len__(self): return self.n
    def __getitem__(self, idx: int): return self.x[idx], self.y[idx]


class ProductTreeDataset(Dataset):
    def __init__(
        self,
        n: int,
        d: int,
        k: int,
        r: int = 2,
        depth: int | None = None,
        device: torch.device | None = None,
        noise_std: float = 0.0,
        seed: int = 0,
        input_dist: str = "uniform",
        indices: List[int] | None = None,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if depth is not None:
            max_k = (r ** depth)
            if k > max_k:
                raise AssertionError(f"k={k} exceeds r**depth={max_k}")
        assert k <= d, f"k={k} must be <= dim={d}"
        self.x = generate_inputs_continuous(n, d, device, seed=seed, dist=input_dist)
        self.y = compute_product_tree_labels(self.x, k=k, r=r, d=depth, indices=indices)
        if noise_std > 0:
            self.y = self.y + noise_std * torch.randn_like(self.y)
        self.n = n
    def __len__(self): return self.n
    def __getitem__(self, idx: int): return self.x[idx], self.y[idx]


class MNISTDataset(Dataset):
    """
    MNIST dataset wrapped to match the interface of other datasets.
    Flattens 28x28 images to 784-dim vectors and converts labels to float regression targets.
    """
    def __init__(
        self,
        n: int,
        d: int,  # Should be 784 for MNIST (28*28), but we'll use it if provided
        spec: str,  # Not used for MNIST, but kept for interface compatibility
        device: torch.device,
        noise_std: float = 0.0,
        seed: int = 0,
        train: bool = True,
    ):
        _ = spec  # Not used for MNIST
        self.device = device
        self.n = n
        
        # Load MNIST data
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
        ])
        
        # Use a fixed seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
        
        mnist_dataset = datasets.MNIST(
            root="./data",
            train=train,
            download=True,
            transform=transform
        )
        
        # Sample n examples (or use all if n is larger than dataset size)
        if n > len(mnist_dataset):
            indices = torch.arange(len(mnist_dataset))
        else:
            g = torch.Generator()
            g.manual_seed(seed)
            indices = torch.randperm(len(mnist_dataset), generator=g)[:n]
        
        # Load data
        images = []
        labels = []
        for idx in indices:
            img, label = mnist_dataset[int(idx)]
            images.append(img.flatten())  # Flatten 28x28 to 784
            labels.append(float(label))  # Convert to float for regression
        
        self.x = torch.stack(images).to(device=device, dtype=torch.float32)
        self.y = torch.tensor(labels, device=device, dtype=torch.float32).unsqueeze(1)
        
        # Handle dimension mismatch: if d != 784, we'll pad or truncate
        if d != 784:
            if d > 784:
                # Pad with zeros
                padding = torch.zeros(self.x.shape[0], d - 784, device=device, dtype=torch.float32)
                self.x = torch.cat([self.x, padding], dim=1)
            else:
                # Truncate
                self.x = self.x[:, :d]
        
        if noise_std > 0:
            g = torch.Generator(device=device)
            g.manual_seed(seed)
            self.y = self.y + noise_std * torch.randn_like(self.y, generator=g)
        
        self.n = len(self.x)
    
    def __len__(self): return self.n
    def __getitem__(self, idx: int): return self.x[idx], self.y[idx]


class RotMNISTDataset(Dataset):
    """
    Rotated MNIST dataset. Same as MNIST but with random rotations applied.
    Rotations are deterministic based on seed for reproducibility.
    """
    def __init__(
        self,
        n: int,
        d: int,  # Should be 784 for MNIST (28*28)
        spec: str,  # Not used, but kept for interface compatibility
        device: torch.device,
        noise_std: float = 0.0,
        seed: int = 0,
        train: bool = True,
        rotation_range: Tuple[float, float] = (-45.0, 45.0),  # Rotation range in degrees
    ):
        _ = spec  # Not used for RotMNIST
        self.device = device
        self.n = n
        
        # Load MNIST data as PIL Images (we'll rotate, then convert to tensor and normalize)
        # Use a fixed seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
        
        # Load raw MNIST dataset (we'll process it ourselves)
        mnist_dataset_raw = datasets.MNIST(
            root="./data",
            train=train,
            download=True,
            transform=None  # No transform - we'll handle it manually
        )
        
        # Sample n examples
        if n > len(mnist_dataset_raw):
            indices = torch.arange(len(mnist_dataset_raw))
        else:
            g = torch.Generator()
            g.manual_seed(seed)
            indices = torch.randperm(len(mnist_dataset_raw), generator=g)[:n]
        
        # Load data and apply rotations
        images = []
        labels = []
        
        for i, idx in enumerate(indices):
            img_pil, label = mnist_dataset_raw[int(idx)]
            
            # Apply deterministic rotation based on seed and index
            rot_gen = torch.Generator()
            rot_gen.manual_seed(seed + i)  # Different rotation for each sample, but deterministic
            angle = rotation_range[0] + (rotation_range[1] - rotation_range[0]) * torch.rand(1, generator=rot_gen).item()
            
            # Rotate the PIL image
            img_rotated_pil = rotate(img_pil, angle, interpolation=transforms.InterpolationMode.BILINEAR)
            
            # Convert to tensor and normalize
            img_tensor = transforms.ToTensor()(img_rotated_pil)  # [0, 1]
            img_tensor = transforms.Normalize((0.1307,), (0.3081,))(img_tensor)  # Normalized
            
            images.append(img_tensor.flatten())  # Flatten 1x28x28 to 784
            labels.append(float(label))
        
        self.x = torch.stack(images).to(device=device, dtype=torch.float32)
        self.y = torch.tensor(labels, device=device, dtype=torch.float32).unsqueeze(1)
        
        # Handle dimension mismatch
        if d != 784:
            if d > 784:
                padding = torch.zeros(self.x.shape[0], d - 784, device=device, dtype=torch.float32)
                self.x = torch.cat([self.x, padding], dim=1)
            else:
                self.x = self.x[:, :d]
        
        if noise_std > 0:
            g = torch.Generator(device=device)
            g.manual_seed(seed)
            self.y = self.y + noise_std * torch.randn_like(self.y, generator=g)
        
        self.n = len(self.x)
    
    def __len__(self): return self.n
    def __getitem__(self, idx: int): return self.x[idx], self.y[idx]


class CIFAR10Dataset(Dataset):
    """
    CIFAR-10 dataset wrapped to match the interface of other datasets.
    Flattens 32x32x3 images to 3072-dim vectors and converts labels to float regression targets.
    """
    def __init__(
        self,
        n: int,
        d: int,  # Should be 3072 for CIFAR-10 (32*32*3), but we'll use it if provided
        spec: str,  # Not used for CIFAR-10, but kept for interface compatibility
        device: torch.device,
        noise_std: float = 0.0,
        seed: int = 0,
        train: bool = True,
    ):
        _ = spec  # Not used for CIFAR-10
        self.device = device
        self.n = n

        # Load CIFAR-10 data
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))  # CIFAR-10 mean and std
        ])

        # Use a fixed seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)

        cifar10_dataset = datasets.CIFAR10(
            root="./data",
            train=train,
            download=True,
            transform=transform
        )

        # Sample n examples (or use all if n is larger than dataset size)
        if n > len(cifar10_dataset):
            indices = torch.arange(len(cifar10_dataset))
        else:
            g = torch.Generator()
            g.manual_seed(seed)
            indices = torch.randperm(len(cifar10_dataset), generator=g)[:n]

        # Load data
        images = []
        labels = []
        for idx in indices:
            img, label = cifar10_dataset[int(idx)]
            images.append(img.flatten())  # Flatten 32x32x3 to 3072
            labels.append(float(label))  # Convert to float for regression

        self.x = torch.stack(images).to(device=device, dtype=torch.float32)
        self.y = torch.tensor(labels, device=device, dtype=torch.float32).unsqueeze(1)

        # Handle dimension mismatch: if d != 3072, we'll pad or truncate
        if d != 3072:
            if d > 3072:
                # Pad with zeros
                padding = torch.zeros(self.x.shape[0], d - 3072, device=device, dtype=torch.float32)
                self.x = torch.cat([self.x, padding], dim=1)
            else:
                # Truncate
                self.x = self.x[:, :d]

        if noise_std > 0:
            g = torch.Generator(device=device)
            g.manual_seed(seed)
            self.y = self.y + noise_std * torch.randn_like(self.y, generator=g)

        self.n = len(self.x)

    def __len__(self): return self.n
    def __getitem__(self, idx: int): return self.x[idx], self.y[idx]


class SVHNDataset(Dataset):
    """
    SVHN dataset wrapped to match the interface of other datasets.
    Flattens 32x32x3 images to 3072-dim vectors and converts labels to float regression targets.
    """
    def __init__(
        self,
        n: int,
        d: int,  # Should be 3072 for SVHN (32*32*3), but we'll use it if provided
        spec: str,  # Not used for SVHN, but kept for interface compatibility
        device: torch.device,
        noise_std: float = 0.0,
        seed: int = 0,
        train: bool = True,
    ):
        _ = spec  # Not used for SVHN
        self.device = device
        self.n = n

        # Load SVHN data
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.198, 0.201, 0.197))  # SVHN mean and std
        ])

        # Use a fixed seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)

        split = 'train' if train else 'test'
        svhn_dataset = datasets.SVHN(
            root="./data",
            split=split,
            download=True,
            transform=transform
        )

        # Sample n examples (or use all if n is larger than dataset size)
        if n > len(svhn_dataset):
            indices = torch.arange(len(svhn_dataset))
        else:
            g = torch.Generator()
            g.manual_seed(seed)
            indices = torch.randperm(len(svhn_dataset), generator=g)[:n]

        # Load data
        images = []
        labels = []
        for idx in indices:
            img, label = svhn_dataset[int(idx)]
            images.append(img.flatten())  # Flatten 32x32x3 to 3072
            labels.append(float(label))  # Convert to float for regression

        self.x = torch.stack(images).to(device=device, dtype=torch.float32)
        self.y = torch.tensor(labels, device=device, dtype=torch.float32).unsqueeze(1)

        # Handle dimension mismatch: if d != 3072, we'll pad or truncate
        if d != 3072:
            if d > 3072:
                # Pad with zeros
                padding = torch.zeros(self.x.shape[0], d - 3072, device=device, dtype=torch.float32)
                self.x = torch.cat([self.x, padding], dim=1)
            else:
                # Truncate
                self.x = self.x[:, :d]

        if noise_std > 0:
            g = torch.Generator(device=device)
            g.manual_seed(seed)
            self.y = self.y + noise_std * torch.randn_like(self.y, generator=g)

        self.n = len(self.x)

    def __len__(self): return self.n
    def __getitem__(self, idx: int): return self.x[idx], self.y[idx]