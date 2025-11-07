
from __future__ import annotations
from typing import List, Dict, Tuple
import numpy as np
import torch
from .models import MLP

@torch.no_grad()
def forward_backward_features(model: MLP, X: torch.Tensor, y: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
    model.eval()
    cache = model.forward_with_cache(X)
    f = cache["f"].squeeze(-1)  # [N]
    # BCEWithLogits gradient; for regression change to (f - y.squeeze(-1))
    delta_out = torch.sigmoid(f) - y.squeeze(-1)

    deltas = [None] * model.depth
    # top layer
    a_L = cache["a"][model.depth - 1]
    phi_prime_L = model.activation_derivative(a_L)
    dL_dhL = delta_out.unsqueeze(-1) * model.out.weight  # [N, width]
    deltas[model.depth - 1] = dL_dhL * phi_prime_L
    # hidden
    for l in reversed(range(model.depth - 1)):
        W_next = model.hidden[l + 1].weight
        a_l = cache["a"][l]
        phi_prime = model.activation_derivative(a_l)
        d_next = deltas[l + 1]
        deltas[l] = torch.matmul(d_next, W_next) * phi_prime

    h_list = [cache["h0"]] + cache["h"]
    return {"h_list": [h for h in h_list], "delta_list": [d for d in deltas], "logits": f}

@torch.no_grad()
def gram_from_features(F: torch.Tensor, normalize_by_n: bool = False) -> torch.Tensor:
    # (optionally) 1/n scaling to match empirical inner product conventions
    if F.dtype in (torch.bfloat16, torch.float16):
        F = F.to(torch.float32)
    G = F @ F.t()
    if normalize_by_n:
        G = G / float(F.shape[0])
    return G

@torch.no_grad()
def layerwise_grams(model: MLP, X: torch.Tensor, y: torch.Tensor, normalize_by_n: bool = True):
    feats = forward_backward_features(model, X, y)
    h_list = feats["h_list"]
    delta_list = feats["delta_list"]
    K_act = [gram_from_features(h, normalize_by_n) for h in h_list]
    K_bp  = [gram_from_features(d, normalize_by_n) for d in delta_list]
    return K_act, K_bp

@torch.no_grad()
def last_hidden_kernel(H_L: torch.Tensor) -> torch.Tensor:
    N = H_L.shape[0]
    K = (H_L @ H_L.t()) / float(N)
    return K.to(torch.float32)

@torch.no_grad()
def eigendecompose_symmetric(K: torch.Tensor, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    # Stay on GPU if possible; upcast to float64 for stability
    Kd = K.detach().to(torch.float64)
    evals, evecs = torch.linalg.eigh(Kd)  # ascending
    evals = torch.flip(evals, dims=[0]).contiguous()
    evecs = torch.flip(evecs, dims=[1]).contiguous()
    if top_k < evals.numel():
        evals = evals[:top_k]
        evecs = evecs[:, :top_k]
    return evals.cpu().numpy().astype(np.float32), evecs.cpu().numpy().astype(np.float32)
