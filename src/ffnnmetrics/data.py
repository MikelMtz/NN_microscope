from __future__ import annotations
import os
from typing import List, Tuple
import torch
from torch.utils.data import Dataset

__all__ = ["parse_interaction_spec", "generate_inputs", "compute_labels", "StaircaseDataset"]

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
