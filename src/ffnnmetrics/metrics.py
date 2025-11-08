from __future__ import annotations
import os, csv
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from torch import nn
# add near the top of metrics.py
import math
from collections import defaultdict


from .kernels import (
    collect_forward_state,
    feature_gram,
    eigendecompose_symmetric,
    layer_backprop_deltas,
    layerwise_feature_grams,
)


def _build_left_eigfuncs(H: torch.Tensor, V_np: np.ndarray, evals_np: np.ndarray, top_k: int) -> torch.Tensor:
    """
    Returns U[:, :k] \in R^{N x k}  with columns ψ_q (left eigenfunctions of K=(1/N) H H^T).
    """
    device, dtype = H.device, H.dtype
    k = min(top_k, V_np.shape[1], evals_np.shape[0])
    if k == 0:
        return torch.zeros((H.shape[0], 0), device=device, dtype=dtype)
    V = torch.from_numpy(V_np[:, :k]).to(device=device, dtype=dtype)  # [m,k]
    lam = torch.from_numpy(evals_np[:k]).to(device=device, dtype=torch.float64)
    lam_inv_sqrt = torch.zeros_like(lam)
    mask = lam > 1e-12
    lam_inv_sqrt[mask] = lam[mask].rsqrt()
    V_scaled = V * lam_inv_sqrt.to(dtype=dtype).unsqueeze(0)
    U = (1.0 / float(np.sqrt(H.shape[0]))) * (H @ V_scaled)            # [N,k]
    return U


# ---------- utils ----------

def _to_cpu_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()

# ---------- scalars (feature-space) ----------

def effective_dimension(C: torch.Tensor) -> float:
    Cd = C.to(torch.float64)
    trC = torch.trace(Cd).item()
    trC2 = (Cd * Cd).sum().item()
    return float(trC2 / (trC**2 + 1e-12))

def Ak_cumulative_feature(V_top_np: np.ndarray, evals_np: np.ndarray, H_np: np.ndarray, y_np: np.ndarray, k: int) -> float:
    V_k = V_top_np[:, :k]              # [m, k]
    lam = evals_np[:k]                 # [k]
    t = H_np.T @ y_np.reshape(-1)      # [m]
    sigma_inv = 1.0 / np.sqrt(np.maximum(lam * float(H_np.shape[0]), 1e-12))  # [k]
    z = (V_k.T @ t) * sigma_inv        # [k]
    num = float(np.dot(z, z))
    den = float(np.dot(y_np.reshape(-1), y_np.reshape(-1)) + 1e-12)
    return num / den

def kdot_speed(C_prev: Optional[torch.Tensor], C_curr: torch.Tensor) -> Tuple[Optional[torch.Tensor], float]:
    if C_prev is None:
        return None, 0.0
    Cdot = (C_curr.to(torch.float64) - C_prev.to(torch.float64))
    num = torch.linalg.norm(Cdot).item()
    den = torch.linalg.norm(C_curr.to(torch.float64)).item() + 1e-12
    return Cdot.to(torch.float32), float(num / den)

@torch.no_grad()
def rotation_speed_rho_k(V_np: np.ndarray, evals_np: np.ndarray, Cdot: Optional[torch.Tensor], k: int) -> float:
    if Cdot is None:
        return 0.0
    Cd = Cdot.detach().to(torch.float64)
    device = Cd.device
    V  = torch.from_numpy(V_np[:, :k]).to(device=device, dtype=torch.float64)  # [m, k]
    M = V.t() @ (Cd @ V)                               # [k,k]
    lam = np.asarray(evals_np[:k], dtype=np.float64)
    gaps = np.abs(lam.reshape(-1, 1) - lam.reshape(1, -1)) + 1e-12
    mask = ~np.eye(k, dtype=bool)
    M_np = M.cpu().numpy()
    num = (M_np[mask] ** 2) / (gaps[mask] ** 2)
    return float(np.sqrt(np.sum(num)))

@torch.no_grad()
def projector_drift(V0_np: np.ndarray, V_np: np.ndarray, k: int) -> float:
    V0 = torch.from_numpy(V0_np[:, :k]).to(torch.float64)
    Vc = torch.from_numpy(V_np[:, :k]).to(torch.float64)
    P0 = V0 @ V0.t(); Pt = Vc @ Vc.t()
    return float(torch.linalg.norm(Pt - P0).item())

# ---------- exact feature-space transfers (finite dataset) ----------

@torch.no_grad()
def compute_feature_transfers_pathwise(
    f_state: Dict[str, List[torch.Tensor]],
    e_vec: torch.Tensor,               # [N]
    V_L_np: np.ndarray,                # [m_L, kL]  (last-layer eigenvectors of C^{(L)})
    evals_L_np: np.ndarray,            # [kL]       (last-layer eigenvalues)
    evals_act_list: List[np.ndarray],  # per-layer eigenvalues of C^{(\ell)}, ℓ=0..L
    V_act_list: List[np.ndarray],      # per-layer eigenvectors of C^{(\ell)}, ℓ=0..L
    top_k: int,                        # kL (last-layer top-k)
    p_top_k: Optional[int] = None,     # number of lower-layer modes p to include (default=top_k)
    chunk: int = 4096,
) -> Tuple[
    List[torch.Tensor],                # T_list (aggregated per layer)  [m_L, m_L]
    List[np.ndarray],                  # diag_by_layer_path[p_k, kL] per layer ℓ=1..L
    List[torch.Tensor],                # B_layers = V^T T^{(ℓ)} V  [kL,kL]
    np.ndarray,                        # topk_inflow [L, kL] (column-sum of B_layers)
    List[float],                       # rotation inflow 𝓡^{(ℓ)}
    float,                             # Δ_scale total
    float                              # Δ_rot total
]:
    """
    Decomposes last-layer Gram time-derivative into layerwise and *pathwise* pieces:
        T^{(ℓ)} = -(2/N) * Sym( sum_p α_p^{(ℓ-1)} M_p^{(ℓ)} )
    where α_p^{(ℓ-1)} = <e, ψ_p^{(ℓ-1)}>,  r_p^{(ℓ)} = (1/N) Δ_ℓ^T ψ_p^{(ℓ-1)},
          and M_p^{(ℓ)} = (1/N) ∑_i h_L(x_i) [ S^{(ℓ→L)}(x_i)^T ( D_ℓ(x_i) r_p^{(ℓ)} ) ]^T.

    Returns per-layer matrices T^{(ℓ)}, per-pathway diagonal inflows into top-k modes,
    V^T T^{(ℓ)} V (for rotation), column inflows, rotation inflow magnitudes, and
    the global rotation-dominance denominators.
    """
    h_list: List[torch.Tensor] = f_state["h_list"]
    D_list: List[torch.Tensor] = f_state["D_list"]
    W_list: List[torch.Tensor] = f_state["W_list"]
    a_vec:  torch.Tensor       = f_state["a_vec"]

    device = h_list[-1].device
    dtype  = h_list[-1].dtype
    N      = h_list[-1].shape[0]
    L      = len(W_list)
    H_L    = h_list[-1].to(device=device, dtype=dtype)   # [N, m_L]
    m_L    = H_L.shape[1]

    # Architectural deltas δ^{(ℓ)}(x) (no labels)
    delta_list = layer_backprop_deltas(W_list, D_list, a_vec)  # [None, δ^1..δ^L]

    # Last-layer top-k basis & gaps
    kL = min(top_k, V_L_np.shape[1])
    Vt = torch.from_numpy(V_L_np[:, :kL]).to(device=device, dtype=dtype)  # [m_L, kL]
    lamL = torch.from_numpy(evals_L_np[:kL]).to(device=device, dtype=torch.float64)
    gaps = torch.abs(lamL[:, None] - lamL[None, :]) + 1e-12  # [kL,kL]

    T_list: List[torch.Tensor]   = []
    B_layers: List[torch.Tensor] = []
    diag_path_by_layer: List[np.ndarray] = []
    topk_inflow = np.zeros((L, kL), dtype=np.float32)

    R_layers: List[float] = []
    Delta_scale_total = 0.0
    Delta_rot_total   = 0.0

    for ell in range(1, L + 1):
        # Left eigenvectors ψ_p^{(ℓ-1)} via H_{ℓ-1}, V^{(ℓ-1)}, Λ^{(ℓ-1)}
        H_prev = h_list[ell - 1].to(device=device, dtype=dtype)            # [N, m_{ℓ-1}]
        evals_prev_np = evals_act_list[ell - 1]
        V_prev_np     = V_act_list[ell - 1]
        if evals_prev_np is None or V_prev_np is None or evals_prev_np.size == 0:
            # Should not happen, but keep a safe guard
            T_list.append(torch.zeros((m_L, m_L), device=device, dtype=dtype))
            B_layers.append(torch.zeros((kL, kL), device=device, dtype=dtype))
            diag_path_by_layer.append(np.zeros((0, kL), dtype=np.float32))
            continue

        P = min(p_top_k if p_top_k is not None else top_k, evals_prev_np.shape[0])
        lam_prev = torch.from_numpy(evals_prev_np[:P]).to(device=device, dtype=torch.float64)
        V_prev   = torch.from_numpy(V_prev_np[:, :P]).to(device=device, dtype=dtype)  # [m_{ℓ-1}, P]

        # Build U^{(ℓ-1)} = (1/√N) * H_{ℓ-1} V_prev Λ_prev^{-1/2}
        lam_mask = (lam_prev > 1e-12)
        lam_inv_sqrt = torch.zeros_like(lam_prev, dtype=torch.float64)
        lam_inv_sqrt[lam_mask] = lam_prev[lam_mask].rsqrt()
        V_scaled = V_prev * lam_inv_sqrt.to(dtype=dtype).unsqueeze(0)       # scale columns
        U_prev  = (1.0 / float(np.sqrt(N))) * (H_prev @ V_scaled)           # [N, P]

        # α_p = <e, ψ_p>,  r_p^{(ℓ)} = (1/N) Δ_ℓ^T ψ_p
        e_col   = e_vec.view(-1, 1).to(device=device, dtype=dtype)          # [N,1]
        alpha   = (U_prev.t() @ e_col).view(-1)                              # [P]
        Delta_l = delta_list[ell].to(device=device, dtype=dtype)             # [N, m_ℓ]
        r_mat   = (Delta_l.t() @ U_prev) / float(N)                          # [m_ℓ, P]

        # Accumulate M_total = ∑_p α_p M_p   (all on device)
        M_total = torch.zeros((m_L, m_L), device=device, dtype=dtype)
        diag_path = torch.zeros((P, kL), device=device, dtype=dtype)

        for p in range(P):
            if not torch.isfinite(alpha[p]):
                continue
            v0 = r_mat[:, p]  # [m_ℓ]
            if torch.all(v0 == 0):
                continue

            M_p = torch.zeros((m_L, m_L), device=device, dtype=dtype)

            # Batch over samples to build M_p = (1/N)∑ H_L(x)^T Q_p(x)
            for s in range(0, N, min(chunk, N)):
                eidx = min(s + chunk, N); B = eidx - s

                Q = D_list[ell - 1][s:eidx, :] * v0.unsqueeze(0).expand(B, -1)  # [B,m_ℓ]
                for r in range(ell + 1, L + 1):
                    W_r = W_list[r - 1].to(device=device, dtype=dtype)
                    Q = Q @ W_r.t()                                             # [B, m_r]
                    Q = D_list[r - 1][s:eidx, :] * Q                            # elem-wise

                H_batch = H_L[s:eidx, :]                                        # [B, m_L]
                M_p = M_p + (H_batch.t() @ Q) / float(N)                        # [m_L, m_L]

            # v_i^T M_p v_i  for top-k last-layer modes  (no need to symmetrize here)
            Z = M_p @ Vt                          # [m_L, kL]
            diag_VMV = torch.sum(Vt * Z, dim=0)   # [kL]
            # Pathway diagonal inflow into top-k modes:
            diag_path[p, :] = (-(2.0 / float(N)) * alpha[p].to(dtype)) * diag_VMV

            # Aggregate M_total for T^{(ℓ)}
            M_total = M_total + alpha[p].to(dtype) * M_p

        # Aggregated per-layer transfer matrix and its projection
        T_layer = -(2.0 / float(N)) * 0.5 * (M_total + M_total.t())           # [m_L, m_L]
        B_layer = Vt.t() @ (T_layer @ Vt)                                      # [kL, kL]

        # Save outputs
        T_list.append(T_layer.to(torch.float32))
        B_layers.append(B_layer.to(torch.float32))
        diag_path_by_layer.append(diag_path.detach().cpu().numpy().astype(np.float32))
        topk_inflow[ell - 1, :] = B_layer.sum(dim=0).detach().cpu().numpy().astype(np.float32)

        # Rotation inflow 𝓡^{(ℓ)} and dominance components
        B = B_layer.to(torch.float64)
        off = B - torch.diag(torch.diag(B))
        R_l = torch.sum(torch.abs(off) / gaps).item()
        R_layers.append(R_l)

        Delta_scale_total += torch.sum(torch.abs(torch.diag(B))).item()
        Delta_rot_total   += torch.sum(torch.abs(off) / gaps).item()

    return T_list, diag_path_by_layer, B_layers, topk_inflow, R_layers, Delta_scale_total, Delta_rot_total




@torch.no_grad()
def compute_feature_transfers_full(
    f_state: Dict[str, List[torch.Tensor]],
    e_vec: torch.Tensor,  # [N]
    chunk: int = 4096,
) -> List[torch.Tensor]:
    """
    Compute exact per-layer transfer matrices T^{(ell)} in feature space:
      \dot C_L = sum_ell T^{(ell)},  T^{(ell)} = -(2/N) * Sym( M^{(ell)} ),
      M^{(ell)} = (1/N) sum_i h_i [ P^{(ell)}(x_i)^T * delta_bar_e^{(ell)} ]^T,
      delta_bar_e^{(ell)} = (1/N) sum_k e_k * delta_k^{(ell)}.

    Returns:
      T_list: list of length L, each T^{(ell)} is [m_L, m_L] (float32, on same device).
    """
    h_list: List[torch.Tensor] = f_state["h_list"]
    D_list: List[torch.Tensor] = f_state["D_list"]
    W_list: List[torch.Tensor] = f_state["W_list"]
    a_vec:  torch.Tensor       = f_state["a_vec"]

    device = h_list[-1].device
    dtype  = h_list[-1].dtype
    N      = h_list[-1].shape[0]
    L      = len(W_list)
    H_L    = h_list[-1].to(device=device, dtype=dtype)   # [N, m_L]
    m_L    = H_L.shape[1]

    # Architectural deltas δ (no labels), then weight by e to get delta_bar_e
    delta_list = layer_backprop_deltas(W_list, D_list, a_vec)  # [None, δ^1..δ^L]

    delta_bar_e: List[Optional[torch.Tensor]] = [None] * (L + 1)
    e_col = e_vec.view(-1, 1).to(device=device, dtype=dtype)   # [N,1]
    for ell in range(1, L + 1):
        delta_bar_e[ell] = (e_col * delta_list[ell]).mean(dim=0)  # [m_ell]

    T_list: List[torch.Tensor] = []
    for ell in range(1, L + 1):
        v0 = delta_bar_e[ell]                                    # [m_ell]
        M = torch.zeros((m_L, m_L), device=device, dtype=dtype)

        for s in range(0, N, min(chunk, N)):
            e = min(s + chunk, N)
            B = e - s

            # P^{(ell)}(x)^T * delta_bar_e^{(ell)} = S^{(ell->L)}(x)^T (D_ell(x) * v0)
            Q = (D_list[ell - 1][s:e, :] * v0.unsqueeze(0).expand(B, -1))  # [B, m_ell]
            for r in range(ell + 1, L + 1):
                W_r = W_list[r - 1].to(device=device, dtype=dtype)
                Q = Q @ W_r.t()                                           # [B, m_r]
                Q = D_list[r - 1][s:e, :] * Q

            H_batch = H_L[s:e, :]                                        # [B, m_L]
            M = M + (H_batch.t() @ Q) / float(N)

        T_ell = -(2.0 / float(N)) * 0.5 * (M + M.t())
        T_list.append(T_ell.to(torch.float32))

    return T_list









##new


def _compute_alpha(H_L: torch.Tensor, V_L_np: np.ndarray, evals_L_np: np.ndarray, y: torch.Tensor, kL: int) -> torch.Tensor:
    U = _build_left_eigfuncs(H_L, V_L_np, evals_L_np, kL)  # [N,k]
    return (U.t() @ y.view(-1,1)).view(-1)                 # [k]

def metric_LARD(B_layers: List[torch.Tensor], gaps: torch.Tensor, alpha: torch.Tensor) -> float:
    """
    LARD = (sum_ell sum_{i!=j} |B_{ji}^{(ell)}| |α_j α_i| / gap_ij) / (sum_ell sum_i |B_{ii}^{(ell)}| α_i^2)
    """
    num = 0.0; den = 0.0
    alpha_abs = torch.abs(alpha).to(torch.float64)
    alpha_sq  = (alpha**2).to(torch.float64)
    for B in B_layers:
        B = B.to(torch.float64)
        k = B.shape[0]
        diag = torch.diag(B)
        den += torch.sum(torch.abs(diag) * alpha_sq[:k]).item()
        off = B - torch.diag(diag)
        # broadcast α_j α_i
        a_outer = (alpha_abs[:k].unsqueeze(1) * alpha_abs[:k].unsqueeze(0))
        num += torch.sum(torch.abs(off) * a_outer / gaps[:k,:k]).item()
    return float(num / (den + 1e-12))

def metric_GSI(B_layers: List[torch.Tensor], gaps: torch.Tensor) -> float:
    gsi = 0.0
    for B in B_layers:
        B = B.to(torch.float64)
        diag = torch.diag(B)
        off = B - torch.diag(diag)
        gsi += torch.sum(torch.abs(off) / (gaps**2)).item()
    return float(gsi)

def metric_pathway_entropy(diag_path_by_layer: List[np.ndarray]) -> Tuple[float, float, float]:
    """
    Returns (Shannon entropy H, Top-5% mass, Gini) over all routes (ℓ,p,i) using |T_{p->i}^{(ℓ)}|.
    """
    if len(diag_path_by_layer) == 0:
        return 0.0, 0.0, 0.0
    w = np.concatenate([np.abs(x).reshape(-1) for x in diag_path_by_layer if x.size > 0], axis=0)
    S = float(w.sum())
    if S <= 0:
        return 0.0, 0.0, 0.0
    p = w / S
    # entropy
    H = float(-(p * (np.log(p + 1e-12))).sum())
    # top-5% mass
    k = max(1, int(0.05 * p.size))
    top_mass = float(np.sort(p)[-k:].sum())
    # Gini
    ps = np.sort(p)
    n = ps.size
    cum = np.cumsum(ps)
    gini = 1.0 - 2.0 * float(np.sum(cum) / (n * np.sum(ps))) + 1.0 / n
    return H, top_mass, gini

def metric_STB_per_layer(diag_path_by_layer: List[np.ndarray]) -> List[np.ndarray]:
    """
    For each layer ℓ, returns STB_i^{(ℓ)} over i=1..kL:
      STB_i = sum_p T_{p->i} / sum_p |T_{p->i}|
    """
    out = []
    for diag_path in diag_path_by_layer:
        if diag_path.size == 0:
            out.append(np.zeros((0,), dtype=np.float32))
            continue
        num = diag_path.sum(axis=0)               # [kL]
        den = np.abs(diag_path).sum(axis=0) + 1e-12
        stb = (num / den).astype(np.float32)
        out.append(stb)
    return out

def metric_STR(prev_evals: Optional[np.ndarray], curr_evals: np.ndarray, k: int) -> float:
    """
    Spectral Turnover Rate: normalized number of pairwise order flips in top-k eigenvalues (descending).
    """
    if prev_evals is None or k <= 1:
        return 0.0
    # ranks by sorting (descending)
    idx_prev = np.argsort(-prev_evals[:k])
    idx_curr = np.argsort(-curr_evals[:k])
    # map previous order positions
    pos_prev = {idx_prev[i]: i for i in range(k)}
    order_prev = [pos_prev[i] for i in range(k)]  # 0..k-1
    # express current as permutation of indices 0..k-1 in prev's label space
    perm = [order_prev[i] for i in idx_curr]
    # count inversions in perm
    inv = 0
    bit = [0]*(k+2)
    def _add(x):
        while x < len(bit):
            bit[x] += 1; x += x & -x
    def _sum(x):
        s = 0
        while x > 0:
            s += bit[x]; x -= x & -x
        return s
    # 1-index for BIT
    for x in [p+1 for p in perm]:
        inv += (_sum(k+1) - _sum(x))
        _add(x)
    norm = k*(k-1)/2.0
    return float(inv / (norm + 1e-12))

@torch.no_grad()
@torch.no_grad()
def compute_interlayer_transfers_pathwise(
    f_state: Dict[str, List[torch.Tensor]],
    e_vec: torch.Tensor,                    # [N]
    evals_act_list: List[np.ndarray],       # eigvals of C^{(ℓ)} for ℓ=0..L
    V_act_list: List[np.ndarray],           # eigvecs of C^{(ℓ)} for ℓ=0..L
    top_k: int,
    p_top_k: Optional[int] = None,
    chunk: int = 4096,
):
    """
    Inter-layer, pathwise transfer decomposition to ANY target layer r (1..L).

    Returns dictionaries keyed by (ell, r):
      - T_pairs[(ell,r)]          : torch.Tensor [m_r, m_r]
      - B_pairs[(ell,r)]          : torch.Tensor [k_r, k_r] with k_r = min(top_k, rank(C^{(r)}))
      - diag_path_pairs[(ell,r)]  : np.ndarray [P_(ell-1) x k_r] of pathwise inflows into top-k of layer r
      - inflow_pairs[(ell,r)]     : np.ndarray [k_r] column sums of B_pairs
      - R_pairs[(ell,r)]          : float  rotation inflow  Σ_{i≠j} |B_{ji}|/gap_ij
      - IGSI_pairs[(ell,r)]       : float  gap-stress       Σ_{i≠j} |B_{ji}|/gap_ij^2
    """
    h_list: List[torch.Tensor] = f_state["h_list"]
    D_list: List[torch.Tensor] = f_state["D_list"]
    W_list: List[torch.Tensor] = f_state["W_list"]
    a_vec:  torch.Tensor       = f_state["a_vec"]

    device = h_list[-1].device
    dtype  = h_list[-1].dtype
    N      = h_list[-1].shape[0]
    L      = len(W_list)

    # architectural (label-free) deltas δ
    delta_list = layer_backprop_deltas(W_list, D_list, a_vec)   # [None, δ^1..δ^L]

    T_pairs, B_pairs = {}, {}
    diag_path_pairs, inflow_pairs = {}, {}
    R_pairs, IGSI_pairs = {}, {}

    for r in range(1, L + 1):
        H_r = h_list[r].to(device=device, dtype=dtype)     # [N, m_r]
        m_r = H_r.shape[1]
        evals_r_np = evals_act_list[r]
        V_r_np     = V_act_list[r]
        if evals_r_np.size == 0 or V_r_np.size == 0:
            continue

        k_r = min(top_k, evals_r_np.shape[0])
        V_r = torch.from_numpy(V_r_np[:, :k_r]).to(device=device, dtype=dtype)   # [m_r, k_r]
        lam_r = torch.from_numpy(evals_r_np[:k_r]).to(device=device, dtype=torch.float64)
        gaps = torch.abs(lam_r[:, None] - lam_r[None, :]) + 1e-12

        for ell in range(1, r + 1):
            # left eigenfuncs ψ_p at layer ell-1 via U_prev = (1/√N) H_{ell-1} V_prev Λ_prev^{-1/2}
            H_prev = h_list[ell - 1].to(device=device, dtype=dtype)     # [N, m_{ell-1}]
            evals_prev_np = evals_act_list[ell - 1]
            V_prev_np     = V_act_list[ell - 1]
            if evals_prev_np.size == 0 or V_prev_np.size == 0:
                continue

            P = min(p_top_k if p_top_k is not None else top_k, evals_prev_np.shape[0])
            lam_prev = torch.from_numpy(evals_prev_np[:P]).to(device=device, dtype=torch.float64)
            V_prev   = torch.from_numpy(V_prev_np[:, :P]).to(device=device, dtype=dtype)
            lam_inv_sqrt = torch.zeros_like(lam_prev)
            mask = lam_prev > 1e-12
            lam_inv_sqrt[mask] = lam_prev[mask].rsqrt()
            V_scaled = V_prev * lam_inv_sqrt.to(dtype=dtype).unsqueeze(0)
            U_prev  = (1.0 / float(np.sqrt(N))) * (H_prev @ V_scaled)           # [N,P]

            # α_p = <e, ψ_p>,  r_p^{(ell)} = (1/N) Δ_ell^T ψ_p
            e_col   = e_vec.view(-1, 1).to(device=device, dtype=dtype)
            alpha   = (U_prev.t() @ e_col).view(-1)                               # [P]
            Delta_l = delta_list[ell].to(device=device, dtype=dtype)              # [N, m_ell]
            r_mat   = (Delta_l.t() @ U_prev) / float(N)                           # [m_ell, P]

            # accumulate M_total and the pathwise diag inflows
            M_total   = torch.zeros((m_r, m_r), device=device, dtype=dtype)
            diag_path = torch.zeros((P, k_r), device=device, dtype=dtype)

            for p in range(P):
                a_p = alpha[p]
                v0  = r_mat[:, p]         # [m_ell]
                if not torch.isfinite(a_p) or torch.all(v0 == 0):
                    continue

                M_p = torch.zeros((m_r, m_r), device=device, dtype=dtype)
                # build M_p = (1/N) ∑ H_r(x)^T Q_p(x) with Q propagated ell->r
                for s in range(0, N, min(chunk, N)):
                    eidx = min(s + chunk, N); B = eidx - s
                    Q = D_list[ell - 1][s:eidx, :] * v0.unsqueeze(0).expand(B, -1)  # [B, m_ell]
                    for rr in range(ell + 1, r + 1):
                        W_rr = W_list[rr - 1].to(device=device, dtype=dtype)
                        Q = Q @ W_rr.t()
                        Q = D_list[rr - 1][s:eidx, :] * Q
                    H_batch = H_r[s:eidx, :]
                    M_p = M_p + (H_batch.t() @ Q) / float(N)

                # pathwise diagonal inflow into top-k at layer r
                Z = M_p @ V_r
                diag_VMV = torch.sum(V_r * Z, dim=0)               # [k_r]
                diag_path[p, :] = (-(2.0 / float(N)) * a_p.to(dtype)) * diag_VMV

                # aggregate
                M_total = M_total + a_p.to(dtype) * M_p

            # per (ell->r) transfer and its top-k projection
            T_pair = -(2.0 / float(N)) * 0.5 * (M_total + M_total.t())
            B_pair = V_r.t() @ (T_pair @ V_r)

            T_pairs[(ell, r)] = T_pair.to(torch.float32)
            B_pairs[(ell, r)] = B_pair.to(torch.float32)
            diag_path_pairs[(ell, r)] = diag_path.detach().cpu().numpy().astype(np.float32)
            inflow_pairs[(ell, r)] = B_pair.sum(dim=0).detach().cpu().numpy().astype(np.float32)

            # rotation metrics for this (ell->r)
            Bf = B_pair.to(torch.float64)
            off = Bf - torch.diag(torch.diag(Bf))
            R_pairs[(ell, r)]   = float(torch.sum(torch.abs(off) / gaps).item())
            IGSI_pairs[(ell, r)] = float(torch.sum(torch.abs(off) / (gaps**2)).item())

    return T_pairs, B_pairs, diag_path_pairs, inflow_pairs, R_pairs, IGSI_pairs




def compute_ILARD_matrix(
    B_pairs: Dict[Tuple[int,int], torch.Tensor],
    evals_act_list: List[np.ndarray],
    V_act_list: List[np.ndarray],
    h_list: List[torch.Tensor],
    y: torch.Tensor,
    top_k: int
) -> Dict[Tuple[int,int], float]:
    out = {}
    for (ell, r), B in B_pairs.items():
        evals_r_np = evals_act_list[r]; V_r_np = V_act_list[r]
        k_r = min(top_k, evals_r_np.shape[0], B.shape[0])
        if k_r == 0:
            out[(ell,r)] = 0.0; continue
        gaps = torch.from_numpy(
            np.abs(evals_r_np[:k_r].reshape(-1,1) - evals_r_np[:k_r].reshape(1,-1)) + 1e-12
        ).to(dtype=torch.float64, device=B.device)
        alpha_r = _compute_alpha(h_list[r], V_r_np, evals_r_np, y, k_r).to(torch.float64).to(B.device)
        a_outer = torch.abs(alpha_r).unsqueeze(1) * torch.abs(alpha_r).unsqueeze(0)
        diag = torch.abs(torch.diag(B[:k_r,:k_r].to(torch.float64))) * (alpha_r[:k_r]**2)
        den = torch.sum(diag).item()
        off = B[:k_r,:k_r].to(torch.float64) - torch.diag(torch.diag(B[:k_r,:k_r].to(torch.float64)))
        num = torch.sum(torch.abs(off) * a_outer / gaps).item()
        out[(ell,r)] = float(num / (den + 1e-12))
    return out

def compute_depth_profiles(B_pairs: Dict[Tuple[int,int], torch.Tensor], evals_act_list: List[np.ndarray], top_k: int) -> Tuple[float, float]:
    num_scale = den_scale = 0.0
    num_rot   = den_rot   = 0.0
    for (ell, r), B in B_pairs.items():
        k_r = min(top_k, B.shape[0])
        lam = evals_act_list[r][:k_r]
        gaps = torch.from_numpy(np.abs(lam.reshape(-1,1) - lam.reshape(1,-1)) + 1e-12).to(B.device, dtype=torch.float64)
        Bk = B[:k_r,:k_r].to(torch.float64)
        d = float(r - ell)
        diag = torch.abs(torch.diag(Bk)).sum().item()
        off  = (torch.abs(Bk - torch.diag(torch.diag(Bk))) / gaps).sum().item()
        num_scale += diag * d
        den_scale += diag
        num_rot   += off * d
        den_rot   += off
    D_scale = float(num_scale / (den_scale + 1e-12))
    D_rot   = float(num_rot   / (den_rot   + 1e-12))
    return D_scale, D_rot

def transported_alignment(
    h_list: List[torch.Tensor],
    D_list: List[torch.Tensor],
    W_list: List[torch.Tensor],
    V_act_list: List[np.ndarray],
    top_k: int,
    chunk: int = 4096
) -> Dict[Tuple[int,int], float]:
    """
    A(ell->r) = (1/k_ell) || V_r^T Sbar^{ell->r} V_ell(:,1:k_ell) ||_F^2
    """
    device = h_list[-1].device
    dtype  = h_list[-1].dtype
    N      = h_list[-1].shape[0]
    L      = len(W_list)
    out = {}
    for r in range(1, L + 1):
        m_r = h_list[r].shape[1]
        V_r_np = V_act_list[r]
        if V_r_np.size == 0:
            continue
        V_r = torch.from_numpy(V_r_np).to(device=device, dtype=dtype)
        for ell in range(1, r + 1):
            V_ell_np = V_act_list[ell]
            if V_ell_np.size == 0:
                out[(ell,r)] = 0.0; continue
            k_ell = min(top_k, V_ell_np.shape[1])
            if k_ell == 0:
                out[(ell,r)] = 0.0; continue
            V_ell = torch.from_numpy(V_ell_np[:, :k_ell]).to(device=device, dtype=dtype)
            # mean sensitivity Sbar^{ell->r}
            Sbar = torch.zeros((m_r, h_list[ell].shape[1]), device=device, dtype=dtype)
            for s in range(0, N, min(chunk, N)):
                eidx = min(s + chunk, N); B = eidx - s
                Q = torch.eye(h_list[ell].shape[1], device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)  # [B,m_ell,m_ell]
                # propagate through layers ell+1..r
                for rr in range(ell + 1, r + 1):
                    W = W_list[rr - 1].to(device=device, dtype=dtype)            # [m_rr, m_{rr-1}]
                    D = D_list[rr - 1][s:eidx, :].unsqueeze(2)                    # [B,m_rr,1]
                    Q = torch.matmul(W, Q.transpose(1,2)).transpose(1,2)         # [B,m_ell,m_rr] -> wrong shape
                    # Correct propagation: we want S^{ell->rr} in [m_rr, m_ell]
                    # We'll build per-sample Jacobian quickly:
                # Simpler, low-memory average: push unit basis columns separately (costly for big widths); skip exact -> proxy:
                # Use linearization: Sbar ≈ E[D_rr.. D_{ell+1}] Π W .
            # For robustness and speed, we approximate by chaining expected diagonals:
            #   E[D] = mean over batch; Sbar ≈ (E[D_{ell+1}] W_{ell+1}) ... (E[D_r] W_r)
            Sbar = torch.eye(h_list[ell].shape[1], device=device, dtype=dtype)
            for rr in range(ell + 1, r + 1):
                Dmean = D_list[rr - 1].mean(dim=0).to(dtype=dtype)              # [m_rr]
                Sbar = torch.matmul(torch.diag(Dmean), W_list[rr - 1].to(dtype=dtype)) @ Sbar  # [m_rr, m_ell]
            # now A = 1/k_ell || V_r^T Sbar V_ell ||_F^2
            A = V_r.t() @ (Sbar @ V_ell)
            val = float((A * A).sum().item() / max(1, k_ell))
            out[(ell,r)] = val
    return out

def interlayer_path_entropy_and_coherence(diag_path_pairs: Dict[Tuple[int,int], np.ndarray]) -> Dict[Tuple[int,int], Tuple[float,float]]:
    out = {}
    for key, arr in diag_path_pairs.items():
        w = np.abs(arr).reshape(-1)
        S = float(w.sum())
        if S <= 0:
            out[key] = (0.0, 0.0); continue
        p = w / S
        H = float(-(p * (np.log(p + 1e-12))).sum())
        coh = float(abs(arr.sum()) / (np.abs(arr).sum() + 1e-12))
        out[key] = (H, coh)
    return out



# ---------- orchestration & logging (feature-space) ----------

@torch.no_grad()
def compute_and_log_all_metrics(
    out_dir: str,
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    top_k: int,
    lower_top_k: int,        # unused (compat)
    epoch: int,
    betas: Tuple[float, float, float],
    track_U: bool,           # unused (compat)
    save_transfers: bool,
    kernel_cache: Dict[str, object],
):
    """
    Feature-space pipeline with:
      - last-layer eigen/Gram analysis
      - pathwise layer transfers (last layer)
      - NEW last-layer advanced metrics: LARD, GSI, RPE, Pathway entropy/Gini/Top5, STB, STR
      - NEW inter-layer transfers to ANY target layer r with ILARD(ell->r), R(ell->r), I-GSI(ell->r),
        depth profiles (D_scale, D_rot), transported alignment, inter-layer path entropy/coherence
      - logging to CSV/NPY/NPZ
    """
    # ---------------- utilities (local) ----------------
    def _build_left_eigfuncs(H: torch.Tensor, V_np: np.ndarray, evals_np: np.ndarray, k: int) -> torch.Tensor:
        device, dtype = H.device, H.dtype
        k = min(k, V_np.shape[1], evals_np.shape[0])
        if k == 0: return torch.zeros((H.shape[0], 0), device=device, dtype=dtype)
        V = torch.from_numpy(V_np[:, :k]).to(device=device, dtype=dtype)
        lam = torch.from_numpy(evals_np[:k]).to(device=device, dtype=torch.float64)
        lam_inv_sqrt = torch.zeros_like(lam); mask = lam > 1e-12; lam_inv_sqrt[mask] = lam[mask].rsqrt()
        V_scaled = V * lam_inv_sqrt.to(dtype=dtype).unsqueeze(0)
        return (1.0 / float(np.sqrt(H.shape[0]))) * (H @ V_scaled)

    def _compute_alpha(H: torch.Tensor, V_np: np.ndarray, evals_np: np.ndarray, y: torch.Tensor, k: int) -> torch.Tensor:
        U = _build_left_eigfuncs(H, V_np, evals_np, k)
        return (U.t() @ y.view(-1,1)).view(-1)  # [k]

    def metric_LARD(B_layers: List[torch.Tensor], gaps: torch.Tensor, alpha: torch.Tensor) -> float:
        num = 0.0; den = 0.0
        alpha = alpha.to(torch.float64)
        a_abs = torch.abs(alpha); a_sq = alpha**2
        for B in B_layers:
            B64 = B.to(torch.float64)
            k   = B64.shape[0]
            diag = torch.diag(B64)
            den += torch.sum(torch.abs(diag) * a_sq[:k]).item()
            off = B64 - torch.diag(diag)
            a_outer = a_abs[:k].unsqueeze(1) * a_abs[:k].unsqueeze(0)
            num += torch.sum(torch.abs(off) * a_outer / gaps[:k,:k]).item()
        return float(num / (den + 1e-12))

    def metric_GSI(B_layers: List[torch.Tensor], gaps: torch.Tensor) -> float:
        gsi = 0.0
        for B in B_layers:
            B64 = B.to(torch.float64)
            off = B64 - torch.diag(torch.diag(B64))
            gsi += torch.sum(torch.abs(off) / (gaps**2)).item()
        return float(gsi)

    def metric_pathway_entropy(diag_path_by_layer: List[np.ndarray]) -> Tuple[float,float,float]:
        w = np.concatenate([np.abs(x).reshape(-1) for x in diag_path_by_layer if x.size > 0], axis=0) if len(diag_path_by_layer)>0 else np.array([])
        S = float(w.sum())
        if S <= 0: return 0.0, 0.0, 0.0
        p = w / S
        H = float(-(p * (np.log(p + 1e-12))).sum())
        k = max(1, int(0.05 * p.size))
        top5 = float(np.sort(p)[-k:].sum())
        # Gini
        ps = np.sort(p); n = ps.size; cum = np.cumsum(ps)
        gini = 1.0 - 2.0 * float(np.sum(cum) / (n * np.sum(ps))) + 1.0 / n
        return H, top5, gini

    def metric_STB_per_layer(diag_path_by_layer: List[np.ndarray]) -> List[np.ndarray]:
        out = []
        for arr in diag_path_by_layer:
            if arr.size == 0: out.append(np.zeros((0,), dtype=np.float32)); continue
            num = arr.sum(axis=0); den = np.abs(arr).sum(axis=0) + 1e-12
            out.append((num/den).astype(np.float32))
        return out

    def metric_STR(prev_evals: Optional[np.ndarray], curr_evals: np.ndarray, k: int) -> float:
        if prev_evals is None or k <= 1: return 0.0
        # simple O(k^2) inversion count between orderings
        prev_order = np.argsort(-prev_evals[:k])
        curr_order = np.argsort(-curr_evals[:k])
        pos_prev = {idx:i for i,idx in enumerate(prev_order)}
        perm = [pos_prev[i] for i in curr_order]
        inv = 0
        for i in range(k):
            for j in range(i+1,k):
                if perm[i] > perm[j]: inv += 1
        return float(inv / (k*(k-1)/2.0 + 1e-12))

    def compute_ILARD_matrix(
        B_pairs: Dict[Tuple[int,int], torch.Tensor],
        evals_act_list: List[np.ndarray],
        V_act_list: List[np.ndarray],
        h_list: List[torch.Tensor],
        y: torch.Tensor,
        top_k: int
    ) -> Dict[Tuple[int,int], float]:
        out = {}
        for (ell, r), B in B_pairs.items():
            evals_r_np = evals_act_list[r]; V_r_np = V_act_list[r]
            k_r = min(top_k, evals_r_np.shape[0], B.shape[0])
            if k_r == 0: out[(ell,r)] = 0.0; continue
            gaps = torch.from_numpy(
                np.abs(evals_r_np[:k_r].reshape(-1,1) - evals_r_np[:k_r].reshape(1,-1)) + 1e-12
            ).to(dtype=torch.float64, device=B.device)
            alpha_r = _compute_alpha(h_list[r], V_r_np, evals_r_np, y, k_r).to(torch.float64).to(B.device)
            a_outer = torch.abs(alpha_r).unsqueeze(1) * torch.abs(alpha_r).unsqueeze(0)
            Bk = B[:k_r,:k_r].to(torch.float64)
            den = torch.sum(torch.abs(torch.diag(Bk)) * (alpha_r[:k_r]**2)).item()
            off = Bk - torch.diag(torch.diag(Bk))
            num = torch.sum(torch.abs(off) * a_outer / gaps).item()
            out[(ell,r)] = float(num / (den + 1e-12))
        return out

    def compute_depth_profiles(B_pairs: Dict[Tuple[int,int], torch.Tensor], evals_act_list: List[np.ndarray], top_k: int) -> Tuple[float,float]:
        num_scale=den_scale=0.0; num_rot=den_rot=0.0
        for (ell, r), B in B_pairs.items():
            k_r = min(top_k, B.shape[0]); 
            if k_r == 0: continue
            lam = evals_act_list[r][:k_r]
            gaps = torch.from_numpy(np.abs(lam.reshape(-1,1) - lam.reshape(1,-1)) + 1e-12).to(B.device, dtype=torch.float64)
            Bk = B[:k_r,:k_r].to(torch.float64)
            d = float(r - ell)
            diag_mass = torch.abs(torch.diag(Bk)).sum().item()
            off_mass  = (torch.abs(Bk - torch.diag(torch.diag(Bk))) / gaps).sum().item()
            num_scale += diag_mass * d; den_scale += diag_mass
            num_rot   += off_mass  * d; den_rot   += off_mass
        return float(num_scale / (den_scale + 1e-12)), float(num_rot / (den_rot + 1e-12))

    def transported_alignment(
        h_list: List[torch.Tensor], D_list: List[torch.Tensor], W_list: List[torch.Tensor],
        V_act_list: List[np.ndarray], top_k: int
    ) -> Dict[Tuple[int,int], float]:
        """
        A(ell->r) = (1/k_ell) || V_r^T Sbar^{ell->r} V_ell(:,1:k_ell) ||_F^2
        Uses fast expectation proxy: Sbar ≈ (E[D_{ell+1}] W_{ell+1}) ... (E[D_r] W_r).
        """
        device = h_list[-1].device; dtype = h_list[-1].dtype; L = len(W_list)
        out = {}
        for r in range(1, L + 1):
            V_r_np = V_act_list[r]; 
            if V_r_np.size == 0: continue
            V_r = torch.from_numpy(V_r_np).to(device=device, dtype=dtype)
            for ell in range(1, r + 1):
                V_ell_np = V_act_list[ell]
                if V_ell_np.size == 0: out[(ell,r)] = 0.0; continue
                k_ell = min(top_k, V_ell_np.shape[1])
                if k_ell == 0: out[(ell,r)] = 0.0; continue
                V_ell = torch.from_numpy(V_ell_np[:, :k_ell]).to(device=device, dtype=dtype)
                # Sbar proxy
                Sbar = torch.eye(h_list[ell].shape[1], device=device, dtype=dtype)
                for rr in range(ell + 1, r + 1):
                    Dmean = D_list[rr - 1].mean(dim=0).to(dtype=dtype)                    # [m_rr]
                    Sbar  = (torch.diag(Dmean) @ W_list[rr - 1].to(dtype=dtype)) @ Sbar   # [m_rr, m_ell]
                A = V_r.t() @ (Sbar @ V_ell)
                out[(ell,r)] = float((A*A).sum().item() / max(1, k_ell))
        return out

    def interlayer_path_entropy_and_coherence(diag_path_pairs: Dict[Tuple[int,int], np.ndarray]) -> Dict[Tuple[int,int], Tuple[float,float]]:
        out = {}
        for key, arr in diag_path_pairs.items():
            w = np.abs(arr).reshape(-1); S = float(w.sum())
            if S <= 0: out[key] = (0.0, 0.0); continue
            p = w / S
            H = float(-(p * (np.log(p + 1e-12))).sum())
            coh = float(abs(arr.sum()) / (np.abs(arr).sum() + 1e-12))
            out[key] = (H, coh)
        return out

    # ---------------- pipeline ----------------
    os.makedirs(out_dir, exist_ok=True)
    device = X.device
    model.eval()

    # Full forward/backward kernels per layer (feature-space)
    C_act_list, C_bp_list, fstate = layerwise_feature_grams(model, X)
    h_list = fstate["h_list"]; D_list = fstate["D_list"]; W_list = fstate["W_list"]

    H_L = h_list[-1]                           # [N, m_L]
    f = fstate["f"].detach().view(-1)          # [N]
    yv = y.view(-1).detach()                   # [N]
    N = X.shape[0]
    L = len(W_list)

    # Per-layer EVDs (activations + backprop)
    V_act_list: List[np.ndarray] = []
    evals_act_list: List[np.ndarray] = []
    V_bp_list: List[Optional[np.ndarray]] = [None]
    evals_bp_list: List[Optional[np.ndarray]] = [None]

    for ell, C_ell in enumerate(C_act_list):  # ℓ=0..L
        k_here = min(top_k, C_ell.shape[0])
        e_act, V_act = eigendecompose_symmetric(C_ell, top_k=k_here)
        evals_act_list.append(e_act); V_act_list.append(V_act)
        np.save(os.path.join(out_dir, f"C_act_eigvals_layer{ell}_epoch{epoch:04d}.npy"), e_act)

    for ell in range(1, L + 1):               # backprop 1..L
        Cb = C_bp_list[ell]
        k_here = min(top_k, Cb.shape[0])
        e_bp, V_bp = eigendecompose_symmetric(Cb, top_k=k_here)
        evals_bp_list.append(e_bp); V_bp_list.append(V_bp)
        np.save(os.path.join(out_dir, f"C_bp_eigvals_layer{ell}_epoch{epoch:04d}.npy"), e_bp)

    # Last layer
    C_L = C_act_list[-1]
    kL  = min(top_k, C_L.shape[0])
    evals_L, V_L = evals_act_list[-1], V_act_list[-1]
    Vt_np = V_L[:, :kL]
    H_L_np = H_L.detach().cpu().numpy()

    # A_k (dual identity)
    Ak = Ak_cumulative_feature(V_L, evals_L, H_L_np, yv.detach().cpu().numpy(), k=kL)

    # Effective dimension (your existing def)
    d_eff = effective_dimension(C_L)

    # Speeds & rotations per layer (for summary)
    if kernel_cache.get("prev_C_act_list") is None:
        kernel_cache["prev_C_act_list"] = [None for _ in range(len(C_act_list))]
    if kernel_cache.get("prev_C_bp_list") is None:
        kernel_cache["prev_C_bp_list"] = [None] + [None for _ in range(L)]

    Cdot_L, dC = kdot_speed(kernel_cache.get("prev_C"), C_L)
    rho_k_last = rotation_speed_rho_k(V_L, evals_L, Cdot_L, k=kL)

    rho_act: List[float] = []
    for ell, (C_curr, V_curr, evals_curr) in enumerate(zip(C_act_list, V_act_list, evals_act_list)):
        Cdot, _ = kdot_speed(kernel_cache["prev_C_act_list"][ell], C_curr)
        k_here = min(top_k, C_curr.shape[0])
        rho_act.append(rotation_speed_rho_k(V_curr, evals_curr, Cdot, k=k_here))

    rho_bp: List[float] = []
    for ell in range(1, L + 1):
        C_curr = C_bp_list[ell]
        V_curr = V_bp_list[ell]
        evals_curr = evals_bp_list[ell]
        Cdot, _ = kdot_speed(kernel_cache["prev_C_bp_list"][ell], C_curr)
        k_here = min(top_k, C_curr.shape[0])
        rho_bp.append(rotation_speed_rho_k(V_curr, evals_curr, Cdot, k=k_here))

    # cache updates
    kernel_cache["prev_C"] = C_L
    kernel_cache["prev_C_act_list"] = C_act_list
    kernel_cache["prev_C_bp_list"]  = C_bp_list

    # Projector drift
    if kernel_cache.get("V0") is None:
        kernel_cache["V0"] = V_L.copy(); Sk = 0.0
    else:
        Sk = projector_drift(kernel_cache["V0"], V_L, k=kL)

    kernel_cache["rho_sum"]   = kernel_cache.get("rho_sum", 0.0) + rho_k_last
    kernel_cache["rho_count"] = kernel_cache.get("rho_count", 0) + 1
    rho_bar = kernel_cache["rho_sum"] / max(1, kernel_cache["rho_count"])

    # ----- Pathwise layer transfers to LAST layer -----
    e_vec = (f - yv).detach()
    (T_list,
     diag_path_by_layer,
     B_layers,
     topk_inflow,
     R_layers,
     Delta_scale_total,
     Delta_rot_total) = compute_feature_transfers_pathwise(
        f_state=fstate,
        e_vec=e_vec,
        V_L_np=Vt_np,
        evals_L_np=evals_L,
        evals_act_list=evals_act_list,
        V_act_list=V_act_list,
        top_k=kL,
        p_top_k=top_k,
        chunk=min(4096, N)
    )

    # shares + global coherence from pathwise |T|
    abs_per_layer, sum_signed, sum_abs = [], 0.0, 0.0
    for diag_path in diag_path_by_layer:
        S_abs = float(np.abs(diag_path).sum()); S_signed = float(diag_path.sum())
        abs_per_layer.append(S_abs); sum_abs += S_abs; sum_signed += S_signed
    shares   = [s / (sum_abs + 1e-12) for s in abs_per_layer] if sum_abs > 0 else [0.0]*len(abs_per_layer)
    C_global = abs(sum_signed) / (sum_abs + 1e-12) if sum_abs > 0 else 0.0
    dom_ratio = Delta_scale_total / (Delta_rot_total + 1e-12)

    # --------- NEW last-layer advanced metrics ----------
    alpha_L = _compute_alpha(H_L, V_L, evals_L, yv, kL)  # [k]
    gaps_L  = torch.from_numpy(np.abs(evals_L[:kL][:,None] - evals_L[:kL][None,:]) + 1e-12).to(device=H_L.device, dtype=torch.float64)

    LARD_val = metric_LARD(B_layers, gaps_L, alpha_L.to(gaps_L.device))
    GSI_val  = metric_GSI(B_layers, gaps_L)

    # RPE
    prev_Ak = kernel_cache.get("prev_Ak", Ak)
    P_prog  = float(Ak - prev_Ak)
    R_y     = 0.0
    for B in B_layers:
        B64  = B.to(torch.float64)
        off  = B64 - torch.diag(torch.diag(B64))
        aout = torch.abs(alpha_L).unsqueeze(1).to(B64.device, dtype=torch.float64) * torch.abs(alpha_L).unsqueeze(0).to(B64.device, dtype=torch.float64)
        R_y += float(torch.sum(torch.abs(off) * aout / gaps_L).item())
    RPE_val = float(P_prog / (R_y + 1e-12))
    kernel_cache["prev_Ak"] = Ak

    # Pathway entropy / Top-5% / Gini
    H_path, top5_mass, gini_mass = metric_pathway_entropy(diag_path_by_layer)
    # STB per layer
    STB_list = metric_STB_per_layer(diag_path_by_layer)
    # STR
    prev_evals = kernel_cache.get("prev_evals_L", None)
    STR_val = metric_STR(prev_evals, evals_L, kL)
    kernel_cache["prev_evals_L"] = evals_L.copy()

    # ---------- NEW inter-layer transfers and metrics ----------
    (T_pairs, B_pairs, diag_path_pairs, inflow_pairs, R_pairs, IGSI_pairs) = compute_interlayer_transfers_pathwise(
        f_state=fstate,
        e_vec=e_vec,
        evals_act_list=evals_act_list,
        V_act_list=V_act_list,
        top_k=top_k,
        p_top_k=top_k,
        chunk=min(4096, N)
    )

    # ILARD_{ell->r} uses y
    ILARD_map = compute_ILARD_matrix(B_pairs, evals_act_list, V_act_list, h_list, yv, top_k)
    # Depth profiles (global)
    D_scale, D_rot = compute_depth_profiles(B_pairs, evals_act_list, top_k=top_k)
    # Transported alignment
    A_transport = transported_alignment(h_list, D_list, W_list, V_act_list, top_k=top_k)
    # Inter-layer path entropy & coherence per (ell->r)
    Hpath_map = interlayer_path_entropy_and_coherence(diag_path_pairs)

    # Composite score (unchanged structure)
    beta1, beta2, beta3 = betas
    G = Ak - beta1 * (d_eff / float(N)) - beta2 * rho_bar - beta3 * (1.0 - C_global)

    # ----------------- Saving -----------------
    # Last-layer eigenpairs (feature space)
    np.save(os.path.join(out_dir, f"K_L_top_eigvals_epoch{epoch:04d}.npy"), evals_L)
    np.save(os.path.join(out_dir, f"K_L_top_eigvecs_epoch{epoch:04d}.npy"), V_L)

    if save_transfers:
        for ell, T_ell in enumerate(T_list, start=1):
            np.save(os.path.join(out_dir, f"T_layer{ell}_epoch{epoch:04d}.npy"),
                    T_ell.detach().cpu().numpy().astype(np.float32))
        np.save(os.path.join(out_dir, f"transfer_inflow_topk_epoch{epoch:04d}.npy"), topk_inflow)
        # inter-layer pathwise diags
        for (ell, r), arr in diag_path_pairs.items():
            np.save(os.path.join(out_dir, f"T_path_diag_{ell}_to_{r}_epoch{epoch:04d}.npy"), arr)

    # Summary CSV (kernel metrics on last layer)
    summ_csv = os.path.join(out_dir, "summary.csv")
    header_exists = os.path.exists(summ_csv) and os.path.getsize(summ_csv) > 0
    with open(summ_csv, "a", newline="") as fcsv:
        w = csv.writer(fcsv)
        if not header_exists:
            header = ["epoch","A_k","d_eff","DeltaC","rho_k","rho_bar","S_k","C","G","dom_ratio"] \
                     + [f"eigval_{i}" for i in range(kL)]
            w.writerow(header)
        w.writerow([epoch, Ak, d_eff, dC, rho_k_last, rho_bar, Sk, C_global, G, dom_ratio]
                   + list(evals_L[:kL]))

    # Shares CSV (from pathwise |T|)
    shares_csv = os.path.join(out_dir, "shares.csv")
    header_exists = os.path.exists(shares_csv) and os.path.getsize(shares_csv) > 0
    with open(shares_csv, "a", newline="") as fsh:
        w = csv.writer(fsh)
        if not header_exists:
            w.writerow(["epoch"] + [f"share_layer_{ell}" for ell in range(1, L + 1)])
        w.writerow([epoch] + shares)

    # Rotation CSVs (per-layer)
    rot_act_csv = os.path.join(out_dir, "rotation_act.csv")
    header_exists = os.path.exists(rot_act_csv) and os.path.getsize(rot_act_csv) > 0
    with open(rot_act_csv, "a", newline="") as fa:
        w = csv.writer(fa)
        if not header_exists:
            w.writerow(["epoch"] + [f"rho_act_layer_{ell}" for ell in range(1, L + 1)])
        w.writerow([epoch] + rho_act[1:])  # skip layer 0

    rot_bp_csv = os.path.join(out_dir, "rotation_bp.csv")
    header_exists = os.path.exists(rot_bp_csv) and os.path.getsize(rot_bp_csv) > 0
    with open(rot_bp_csv, "a", newline="") as fb:
        w = csv.writer(fb)
        if not header_exists:
            w.writerow(["epoch"] + [f"rho_bp_layer_{ell}" for ell in range(1, L + 1)])
        w.writerow([epoch] + rho_bp)

    # Alignment CSV (unchanged header logic; populate with your existing numbers if needed)
    align_csv = os.path.join(out_dir, "alignment.csv")
    header_exists = os.path.exists(align_csv) and os.path.getsize(align_csv) > 0
    with open(align_csv, "a", newline="") as fal:
        w = csv.writer(fal)
        if not header_exists:
            w.writerow(["epoch"] + [f"align_layer_{ell}" for ell in range(1, L + 1)])
        # keep previous behavior or compute here as desired

    # NEW: last-layer advanced metrics CSV
    adv_csv = os.path.join(out_dir, "lastlayer_advanced.csv")
    header_exists = os.path.exists(adv_csv) and os.path.getsize(adv_csv) > 0
    with open(adv_csv, "a", newline="") as fa:
        w = csv.writer(fa)
        if not header_exists:
            w.writerow(["epoch","LARD","GSI","RPE","H_path","Top5_mass","Gini","STR"])
        w.writerow([epoch, LARD_val, GSI_val, RPE_val, H_path, top5_mass, gini_mass, STR_val])

    # save STB arrays (one per layer) for detailed analysis
    for ell, stb in enumerate(STB_list, start=1):
        np.save(os.path.join(out_dir, f"STB_layer{ell}_epoch{epoch:04d}.npy"), stb)

    # ---- NEW: inter-layer summaries ----
    ilard_csv = os.path.join(out_dir, "interlayer_ilard.csv")
    header_exists = os.path.exists(ilard_csv) and os.path.getsize(ilard_csv) > 0
    with open(ilard_csv, "a", newline="") as fi:
        w = csv.writer(fi)
        if not header_exists:
            w.writerow(["epoch","ell","r","ILARD","R_inflow","I_GSI","H_path_pair","Coherence_pair"])
        for (ell, r), val in ILARD_map.items():
            H_pair, C_pair = Hpath_map.get((ell,r), (0.0,0.0))
            w.writerow([epoch, ell, r, val, R_pairs.get((ell,r),0.0), IGSI_pairs.get((ell,r),0.0), H_pair, C_pair])

    # depth profiles
    depth_csv = os.path.join(out_dir, "interlayer_depth_profiles.csv")
    header_exists = os.path.exists(depth_csv) and os.path.getsize(depth_csv) > 0
    with open(depth_csv, "a", newline="") as fd:
        w = csv.writer(fd)
        if not header_exists:
            w.writerow(["epoch","D_scale","D_rot"])
        w.writerow([epoch, D_scale, D_rot])

    # transported alignment
    ta_csv = os.path.join(out_dir, "transported_alignment.csv")
    header_exists = os.path.exists(ta_csv) and os.path.getsize(ta_csv) > 0
    with open(ta_csv, "a", newline="") as ft:
        w = csv.writer(ft)
        if not header_exists:
            w.writerow(["epoch","ell","r","A_transport"])
        for (ell,r), val in A_transport.items():
            w.writerow([epoch, ell, r, val])



@torch.no_grad()
def make_kernel_cache(out_dir: str, k: int) -> Dict[str, object]:
    os.makedirs(out_dir, exist_ok=True)
    return dict(
        prev_C=None,              # last layer only (compat)
        prev_C_act_list=None,     # list over ℓ=0..L
        prev_C_bp_list=None,      # list over ℓ=1..L (index 0 unused)
        V0=None,
        rho_sum=0.0,
        rho_count=0,
        S_sum=0.0,
        top_k=k
    )
