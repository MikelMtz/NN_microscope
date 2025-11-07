
from __future__ import annotations
import os, csv
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from torch import nn
from .kernels import last_hidden_kernel, eigendecompose_symmetric

# ---------- utils ----------

def _to_cpu_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()

# ---------- core scalars ----------

def effective_dimension(K: torch.Tensor) -> float:
    Kd = K.to(torch.float64)
    trK = torch.trace(Kd).item()
    trK2 = (Kd * Kd).sum().item()
    return float(trK2 / (trK**2 + 1e-12))  # user's definition

def Ak_cumulative(U_top_np: np.ndarray, y_np: np.ndarray, k: int) -> float:
    U = U_top_np[:, :k]
    num = float(np.sum((U.T @ y_np.reshape(-1)) ** 2))
    den = float(np.dot(y_np.reshape(-1), y_np.reshape(-1)) + 1e-12)
    return num / den

def kdot_speed(K_prev: Optional[torch.Tensor], K_curr: torch.Tensor) -> Tuple[Optional[torch.Tensor], float]:
    if K_prev is None:
        return None, 0.0
    Kdot = (K_curr.to(torch.float64) - K_prev.to(torch.float64))
    num = torch.linalg.norm(Kdot).item()
    den = torch.linalg.norm(K_curr.to(torch.float64)).item() + 1e-12
    return Kdot.to(torch.float32), float(num / den)

@torch.no_grad()
def rotation_speed_rho_k(U_np: np.ndarray, evals_np: np.ndarray, Kdot: Optional[torch.Tensor], k: int) -> float:
    if Kdot is None:
        return 0.0
    Kd = Kdot.detach().to(torch.float64)            # stays on its current device (GPU if K was GPU)
    device = Kd.device
    U  = torch.from_numpy(U_np[:, :k]).to(device=device, dtype=torch.float64)  # <-- key change
    M = U.t() @ (Kd @ U)                              # [k,k] on same device
    lam = np.asarray(evals_np[:k], dtype=np.float64)
    gaps = np.abs(lam.reshape(-1, 1) - lam.reshape(1, -1)) + 1e-12
    mask = ~np.eye(k, dtype=bool)
    M_np = M.cpu().numpy()
    num = (M_np[mask] ** 2) / (gaps[mask] ** 2)
    return float(np.sqrt(np.sum(num)))

@torch.no_grad()
def projector_drift(U0_np: np.ndarray, U_np: np.ndarray, k: int) -> float:
    U0 = torch.from_numpy(U0_np[:, :k]).to(torch.float64)
    Uc = torch.from_numpy(U_np[:, :k]).to(torch.float64)
    P0 = U0 @ U0.t(); Pt = Uc @ Uc.t()
    return float(torch.linalg.norm(Pt - P0).item())

# ---------- forward state capture (hidden layers) ----------

@torch.no_grad()
@torch.no_grad()
def collect_forward_state(model: nn.Module, X: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
    """
    Supports two layouts:
      (A) Old: model.net = Sequential([... Linear, ReLU ...] ... Linear(out=1))
      (B) New: model.hidden = ModuleList([Linear,...]), model.out = Linear(->1), model.act present
    Returns dict with: h_list, u_list, D_list, W_list, a_vec, f
    D_list uses the model's activation derivative when available (works for ReLU/SiLU/etc).
    """
    device = X.device
    dtype  = X.dtype

    # --- Case B: new MLP API with .hidden / .out ---
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

    # --- Case A: legacy API with .net (Sequential) ---
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
        # assume ReLU in legacy path
        D = (u > 0).to(x.dtype)
        D_list.append(D)
        act = layers[lin_ptr + 1]; assert isinstance(act, nn.ReLU)
        x = torch.relu(u)
        h_list.append(x)
        lin_ptr += 2

    out_lin = layers[lin_ptr]; assert isinstance(out_lin, nn.Linear) and out_lin.out_features == 1
    a_vec = out_lin.weight.squeeze(0)
    b_out = out_lin.bias.squeeze(0) if out_lin.bias is not None else torch.tensor(0., device=device, dtype=x.dtype)
    f = (x @ a_vec) + b_out

    return dict(h_list=h_list, u_list=u_list, D_list=D_list, W_list=W_list, a_vec=a_vec, f=f)



# ---------- transfers (finite-dataset estimator) ----------

@torch.no_grad()
def compute_transfers(
    model: nn.Module,
    X: torch.Tensor,
    f_state: Dict[str, List[torch.Tensor]],
    U_L_np: np.ndarray,
    layer_evects: List[np.ndarray],
    layer_evals: List[np.ndarray],
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    device = X.device; dtype = X.dtype
    N = X.shape[0]
    h_list = f_state["h_list"]; D_list = f_state["D_list"]; W_list = f_state["W_list"]; a_vec = f_state["a_vec"]
    L = len(W_list)

    delta_list: List[torch.Tensor] = [None] * (L + 1)
    delta_L = D_list[-1] * a_vec.to(device=device, dtype=dtype).unsqueeze(0)
    delta_list[L] = delta_L
    for ell in range(L - 1, 0, -1):
        delta_next = delta_list[ell + 1]
        W_next = W_list[ell].to(device=device, dtype=dtype)
        tmp = delta_next @ W_next
        delta_list[ell] = tmp * D_list[ell - 1]
    delta_bar = [None] + [delta_list[ell].mean(dim=0) for ell in range(1, L + 1)]

    H_L = h_list[-1].to(device=device, dtype=dtype)
    U_L = torch.from_numpy(U_L_np).to(device=device, dtype=dtype)
    V_right = H_L.t() @ U_L  # [m_L, k]

    P_lower = layer_evects[0].shape[1] if len(layer_evects) > 0 else 0
    T = torch.zeros((L, U_L.shape[1], P_lower), device=device, dtype=dtype)

    chunk = min(N, 4096)
    for ell in range(1, L + 1):
        m_ell = W_list[ell - 1].shape[0]
        U_low_np = layer_evects[ell - 1]
        U_low = torch.from_numpy(U_low_np).to(device=device, dtype=dtype)
        R = torch.zeros((U_L.shape[1], P_lower, m_ell), device=device, dtype=dtype)
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            B = e - s
            Y = V_right.unsqueeze(0).expand(B, -1, -1).clone()
            for r in range(L, ell, -1):
                W_r = W_list[r - 1].to(device=device, dtype=dtype)
                D_r = D_list[r - 1][s:e, :]
                Y = (D_r.unsqueeze(2) * Y).transpose(1, 2) @ W_r
                Y = Y.transpose(1, 2)
            D_ell = D_list[ell - 1][s:e, :]
            Y = D_ell.unsqueeze(2) * Y
            Tmat_T = Y.transpose(1, 2)
            R = R + torch.einsum('bk,bp,bkm->kpm', U_L[s:e, :], U_low[s:e, :], Tmat_T)
        R = R / float(N * N)
        delta_bar_ell = delta_bar[ell]
        T[ell - 1] = torch.einsum('kpm,m->kp', R, delta_bar_ell)

    return T, torch.zeros((L,), device=device, dtype=dtype), 0.0

# ---------- orchestration & logging ----------

@torch.no_grad()
def compute_and_log_all_metrics(
    out_dir: str,
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    top_k: int,
    lower_top_k: int,
    epoch: int,
    betas: Tuple[float, float, float],
    track_U: bool,
    save_transfers: bool,
    kernel_cache: Dict[str, object],
):
    os.makedirs(out_dir, exist_ok=True)
    device = X.device
    model.eval()

    fstate = collect_forward_state(model, X)
    H_L = fstate["h_list"][-1]
    f = fstate["f"].detach()
    yv = y.view(-1).detach()

    K_L = last_hidden_kernel(H_L)
    evals_L, U_L = eigendecompose_symmetric(K_L, top_k=top_k)

    A_k = Ak_cumulative(U_L, _to_cpu_np(yv), k=top_k)
    d_eff = effective_dimension(K_L)
    Kdot, dK = kdot_speed(kernel_cache.get("prev_K"), K_L)
    rho_k = rotation_speed_rho_k(U_L, evals_L, Kdot, k=top_k)

    if kernel_cache.get("U0") is None:
        kernel_cache["U0"] = U_L.copy(); Sk = 0.0
    else:
        Sk = projector_drift(kernel_cache["U0"], U_L, k=top_k)

    kernel_cache["rho_sum"] = kernel_cache.get("rho_sum", 0.0) + rho_k
    kernel_cache["rho_count"] = kernel_cache.get("rho_count", 0) + 1
    rho_bar = kernel_cache["rho_sum"] / max(1, kernel_cache["rho_count"])

    h_list = fstate["h_list"]
    Lh = len(h_list) - 1
    layer_evects: List[np.ndarray] = []
    layer_evals: List[np.ndarray] = []
    for ellm1 in range(0, Lh):
        H_ellm1 = h_list[ellm1]
        K_low = last_hidden_kernel(H_ellm1)
        ev, Uv = eigendecompose_symmetric(K_low, top_k=min(lower_top_k, K_low.shape[0]))
        layer_evals.append(ev)
        layer_evects.append(Uv)

    T_raw, _, _ = compute_transfers(model, X, fstate, U_L_np=U_L, layer_evects=layer_evects, layer_evals=layer_evals)
    e_vec = (f - yv).detach()
    N = X.shape[0]
    coherence_num = 0.0
    T_full = torch.zeros_like(T_raw)
    denom_abs = 0.0
    for ell in range(1, Lh + 1):
        U_low = torch.from_numpy(layer_evects[ell - 1]).to(device=device, dtype=X.dtype)
        lam_low = layer_evals[ell - 1]
        eproj = (U_low.t() @ e_vec.to(U_low.dtype)) / float(N)
        scale = torch.from_numpy(lam_low).to(device=device, dtype=U_low.dtype) * eproj
        T_full[ell - 1] = T_raw[ell - 1] * scale.unsqueeze(0)
        S_signed = torch.sum(T_full[ell - 1])
        S_abs = torch.sum(torch.abs(T_full[ell - 1]))
        coherence_num += float(torch.abs(S_signed).item())
        denom_abs += float(S_abs.item())

    C = (coherence_num / (denom_abs + 1e-12)) if denom_abs > 0 else 0.0
    shares = []
    for ell in range(1, Lh + 1):
        S_abs = torch.sum(torch.abs(T_full[ell - 1])).item()
        shares.append(S_abs)
    S_total = sum(shares) + 1e-12
    shares = [s / S_total for s in shares]

    beta1, beta2, beta3 = betas
    G = A_k - beta1 * (d_eff / float(N)) - beta2 * rho_bar - beta3 * (1.0 - C)

    epoch_dir = os.path.join(out_dir, f"epoch_{epoch:04d}")
    os.makedirs(epoch_dir, exist_ok=True)
    np.save(os.path.join(epoch_dir, "K_L_top_eigvals.npy"), evals_L)
    np.save(os.path.join(epoch_dir, "K_L_top_eigvecs.npy"), U_L)
    if save_transfers:
        np.save(os.path.join(epoch_dir, "transfers.npy"), _to_cpu_np(T_full))

    summ_csv = os.path.join(out_dir, "summary.csv")
    header_exists = os.path.exists(summ_csv) and os.path.getsize(summ_csv) > 0
    with open(summ_csv, "a", newline="") as f:
        w = csv.writer(f)
        if not header_exists:
            header = ["epoch","A_k","d_eff","DeltaK","rho_k","rho_bar","S_k","C","G"] + [f"eigval_{i}" for i in range(top_k)]
            w.writerow(header)
        w.writerow([epoch, A_k, d_eff, dK, rho_k, rho_bar, Sk, C, G] + list(evals_L[:top_k]))

    shares_csv = os.path.join(out_dir, "shares.csv")
    header_exists = os.path.exists(shares_csv) and os.path.getsize(shares_csv) > 0
    with open(shares_csv, "a", newline="") as f:
        w = csv.writer(f)
        if not header_exists:
            w.writerow(["epoch"] + [f"share_layer_{ell}" for ell in range(1, Lh + 1)])
        w.writerow([epoch] + shares)

    kernel_cache["prev_K"] = K_L
    if kernel_cache.get("U0") is None:
        kernel_cache["U0"] = U_L.copy()
    kernel_cache["S_sum"] = kernel_cache.get("S_sum", 0.0) + (0.0 if "Sk" not in locals() else Sk)

@torch.no_grad()
def make_kernel_cache(out_dir: str, k: int) -> Dict[str, object]:
    os.makedirs(out_dir, exist_ok=True)
    return dict(prev_K=None, U0=None, rho_sum=0.0, rho_count=0, S_sum=0.0, top_k=k)
