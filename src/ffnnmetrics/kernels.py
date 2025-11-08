from __future__ import annotations
from typing import List, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from .models import MLP

# ---------------------------
# Core helpers (feature-space)
# ---------------------------

@torch.no_grad()
def feature_gram(F: torch.Tensor, normalize_by_n: bool = True) -> torch.Tensor:
    """
    Neuron–neuron Gram: C = (F^T F) / N if normalize_by_n else (F^T F).
    Shapes:
      F: [N, m]  ->  C: [m, m]
    """
    if F.dtype in (torch.bfloat16, torch.float16):
        F = F.to(torch.float32)
    C = F.t() @ F
    if normalize_by_n:
        C = C / float(F.shape[0])
    return C.to(torch.float32)

@torch.no_grad()
def eigendecompose_symmetric(C: torch.Tensor, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    EVD of symmetric feature Gram (float64 for stability), returns top-k in descending order.
    Returns:
      evals: [top_k]            (np.float32)
      evecs: [m, top_k]         (columns are eigenvectors)  (np.float32)
    """
    Cd = C.detach().to(torch.float64)
    evals, evecs = torch.linalg.eigh(Cd)  # ascending
    evals = torch.flip(evals, dims=[0]).contiguous()
    evecs = torch.flip(evecs, dims=[1]).contiguous()
    if top_k < evals.numel():
        evals = evals[:top_k]
        evecs = evecs[:, :top_k]
    return evals.cpu().numpy().astype(np.float32), evecs.cpu().numpy().astype(np.float32)

# -----------------------------------------
# Forward state (used by metrics & kernels)
# -----------------------------------------

@torch.no_grad()
def collect_forward_state(model: nn.Module, X: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
    """
    Supports:
      (B) model.hidden: ModuleList([Linear,...]), model.out: Linear(->1), model.act, model.activation_derivative
      (A) legacy model.net: Sequential([... Linear, ReLU ...] ... Linear(out=1))
    Returns dict with:
      h_list: [h0=X, h1, ..., hL]        (each [N, m_l])
      u_list: [u1, u2, ..., uL]          (each [N, m_l])
      D_list: [D1, D2, ..., DL]          (each [N, m_l], activation derivative values)
      W_list: [W1, W2, ..., WL]          (each [m_l, m_{l-1}])
      a_vec:  output weight vector [m_L]
      f:      model output [N]
    """
    device = X.device
    dtype  = X.dtype

    if hasattr(model, "hidden") and hasattr(model, "out"):
        act_fn = getattr(model, "act", None)
        act_deriv = getattr(model, "activation_derivative", None)
        assert act_fn is not None and act_deriv is not None, "Model must expose act and activation_derivative"

        W_list: List[torch.Tensor] = [lin.weight for lin in model.hidden]
        h_list: List[torch.Tensor] = [X]
        u_list: List[torch.Tensor] = []
        D_list: List[torch.Tensor] = []

        x = X
        for lin in model.hidden:
            u = x @ lin.weight.t() + (lin.bias if lin.bias is not None else 0.0)
            u_list.append(u)
            D = act_deriv(u)                         # works for ReLU/SiLU/GELU/Tanh
            D_list.append(D.to(dtype))
            x = act_fn(u)
            h_list.append(x)

        a_vec = model.out.weight.squeeze(0)
        b_out = model.out.bias.squeeze(0) if model.out.bias is not None else torch.tensor(0., device=device, dtype=dtype)
        f = (x @ a_vec) + b_out
        return dict(h_list=h_list, u_list=u_list, D_list=D_list, W_list=W_list, a_vec=a_vec, f=f)

    # --- Legacy Sequential path ---
    layers = list(model.net)
    W_list: List[torch.Tensor] = []
    h_list: List[torch.Tensor] = [X]
    u_list: List[torch.Tensor] = []
    D_list: List[torch.Tensor] = []

    lin_indices = [i for i, m in enumerate(layers) if isinstance(m, nn.Linear)]
    L = len(lin_indices) - 1
    assert L >= 1, "Need at least 1 hidden layer."

    x = X
    lin_ptr = 0
    for _ in range(L):
        lin = layers[lin_ptr]; assert isinstance(lin, nn.Linear)
        W_list.append(lin.weight)
        u = x @ lin.weight.t() + (lin.bias if lin.bias is not None else 0.0)
        u_list.append(u)
        D = (u > 0).to(x.dtype)                      # assume ReLU deriv
        D_list.append(D)
        act = layers[lin_ptr + 1]; assert isinstance(act, nn.ReLU)
        x = torch.relu(u)
        h_list.append(x)
        lin_ptr += 2

    out_lin = layers[lin_ptr]; assert isinstance(out_lin, nn.Linear) and out_lin.out_features == 1
    a_vec = out_lin.weight.squeeze(0)
    b_out = out_lin.bias.squeeze(0) if out_lin.bias is not None else torch.tensor(0., device=X.device, dtype=x.dtype)
    f = (x @ a_vec) + b_out
    return dict(h_list=h_list, u_list=u_list, D_list=D_list, W_list=W_list, a_vec=a_vec, f=f)

# -------------------------------------------------------
# Backprop "delta" signals (architecture-only, no labels)
# -------------------------------------------------------

@torch.no_grad()
def layer_backprop_deltas(W_list: List[torch.Tensor], D_list: List[torch.Tensor], a_vec: torch.Tensor) -> List[torch.Tensor]:
    """
    Per-sample architectural backprop signals (no label dependence):
      δ^L = D_L ⊙ a
      δ^ℓ = (δ^{ℓ+1} W_{ℓ+1}) ⊙ D_ℓ
    Returns: [None, δ^1, δ^2, ..., δ^L] where δ^ℓ has shape [N, m_ℓ].
    """
    L = len(W_list)
    dtype = D_list[0].dtype
    device = D_list[0].device

    delta_list: List[torch.Tensor] = [None] * (L + 1)
    delta_L = D_list[-1] * a_vec.to(device=device, dtype=dtype).unsqueeze(0)
    delta_list[L] = delta_L
    for ell in range(L - 1, 0, -1):
        delta_next = delta_list[ell + 1]                 # [N, m_{ell+1}]
        W_next = W_list[ell].to(device=device, dtype=dtype)
        tmp = delta_next @ W_next                        # [N, m_ell]
        delta_list[ell] = tmp * D_list[ell - 1]          # [N, m_ell]
    return delta_list

# ---------------------------------------------------
# Layerwise feature-space forward/backward Gram lists
# ---------------------------------------------------

@torch.no_grad()
def layerwise_feature_grams(model: MLP, X: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor], Dict[str, List[torch.Tensor]]]:
    """
    Returns:
      C_act:  [C^0, C^1, ..., C^L] with C^ℓ = (H_ℓ^T H_ℓ)/N
      C_bp:   [None, C_δ^1, ..., C_δ^L] with C_δ^ℓ = (Δ_ℓ^T Δ_ℓ)/N
      fstate: forward state dict (h_list, D_list, W_list, a_vec, f)
    """
    fstate = collect_forward_state(model, X)
    h_list = fstate["h_list"]
    D_list = fstate["D_list"]
    W_list = fstate["W_list"]
    a_vec  = fstate["a_vec"]

    C_act = [feature_gram(h) for h in h_list]  # ℓ=0..L
    delta_list = layer_backprop_deltas(W_list, D_list, a_vec)
    C_bp  = [None] + [feature_gram(delta_list[ell]) for ell in range(1, len(delta_list))]
    return C_act, C_bp, fstate


