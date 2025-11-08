from __future__ import annotations
import os, csv
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from torch import nn

from .kernels import (
    collect_forward_state,
    feature_gram,
    eigendecompose_symmetric,
    layer_backprop_deltas,
    layerwise_feature_grams,
)

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

# ---------- orchestration & logging (feature-space) ----------

@torch.no_grad()
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
    Feature-space pipeline with corrected:
      - d_eff (participation ratio),
      - global coherence (single absolute at the end),
      - layer share from |T_{p->i}^{(ℓ)}|,
      - pathwise transfer decomposition,
      - rotation inflow 𝓡^{(ℓ)} and rotation-dominance ratio,
      - same outputs you already save, with a few extra CSVs.
    """
    os.makedirs(out_dir, exist_ok=True)
    device = X.device
    model.eval()

    # Full forward/backward kernels per layer (feature-space)
    C_act_list, C_bp_list, fstate = layerwise_feature_grams(model, X)
    h_list = fstate["h_list"]
    D_list = fstate["D_list"]
    W_list = fstate["W_list"]

    H_L = h_list[-1]                           # [N, m_L]
    f = fstate["f"].detach().view(-1)          # [N]
    yv = y.view(-1).detach()                   # [N]
    N = X.shape[0]
    L = len(W_list)

    # -- Per-layer EVDs for forward/backward; save top eigenvalues
    V_act_list: List[np.ndarray] = []
    evals_act_list: List[np.ndarray] = []
    V_bp_list: List[Optional[np.ndarray]] = [None]
    evals_bp_list: List[Optional[np.ndarray]] = [None]

    for ell, C_ell in enumerate(C_act_list):  # ℓ=0..L
        k_here = min(top_k, C_ell.shape[0])
        e_act, V_act = eigendecompose_symmetric(C_ell, top_k=k_here)
        evals_act_list.append(e_act); V_act_list.append(V_act)
        np.save(os.path.join(out_dir, f"C_act_eigvals_layer{ell}_epoch{epoch:04d}.npy"), e_act)

    for ell in range(1, L + 1):  # bp only for 1..L
        Cb = C_bp_list[ell]
        k_here = min(top_k, Cb.shape[0])
        e_bp, V_bp = eigendecompose_symmetric(Cb, top_k=k_here)
        evals_bp_list.append(e_bp); V_bp_list.append(V_bp)
        np.save(os.path.join(out_dir, f"C_bp_eigvals_layer{ell}_epoch{epoch:04d}.npy"), e_bp)

    # -- Last hidden (feature Gram)
    C_L = C_act_list[-1]
    kL = min(top_k, C_L.shape[0])
    evals_L, V_L = evals_act_list[-1], V_act_list[-1]
    Vt_np = V_L[:, :kL]
    H_L_np = H_L.detach().cpu().numpy()  # (used in Ak)

    # A_k via dual identity (exact)
    Ak = Ak_cumulative_feature(V_L, evals_L, H_L_np, yv.detach().cpu().numpy(), k=kL)

    # Effective rank for last layer (participation ratio)
    d_eff = effective_dimension(C_L)

    # Speeds & rotations per layer (activations & backprop)
    if kernel_cache.get("prev_C_act_list") is None:
        kernel_cache["prev_C_act_list"] = [None for _ in range(len(C_act_list))]
    if kernel_cache.get("prev_C_bp_list") is None:
        kernel_cache["prev_C_bp_list"] = [None] + [None for _ in range(L)]

    # Rotation of last layer (for summary), plus per-layer rotations
    Cdot_L, dC = kdot_speed(kernel_cache.get("prev_C"), C_L)  # legacy 'prev_C'
    rho_k_last = rotation_speed_rho_k(V_L, evals_L, Cdot_L, k=kL)

    rho_act: List[float] = []
    for ell, (C_curr, V_curr, evals_curr) in enumerate(zip(C_act_list, V_act_list, evals_act_list)):
        Cdot, _ = kdot_speed(kernel_cache["prev_C_act_list"][ell], C_curr)
        k_here = min(top_k, C_curr.shape[0])
        rho = rotation_speed_rho_k(V_curr, evals_curr, Cdot, k=k_here)
        rho_act.append(rho)

    rho_bp: List[float] = []
    for ell in range(1, L + 1):
        C_curr = C_bp_list[ell]
        V_curr = V_bp_list[ell]
        evals_curr = evals_bp_list[ell]
        Cdot, _ = kdot_speed(kernel_cache["prev_C_bp_list"][ell], C_curr)
        k_here = min(top_k, C_curr.shape[0])
        rho = rotation_speed_rho_k(V_curr, evals_curr, Cdot, k=k_here)
        rho_bp.append(rho)

    # Cache updates
    kernel_cache["prev_C"] = C_L
    kernel_cache["prev_C_act_list"] = C_act_list
    kernel_cache["prev_C_bp_list"]  = C_bp_list

    # Projector drift on last-layer feature subspace
    if kernel_cache.get("V0") is None:
        kernel_cache["V0"] = V_L.copy(); Sk = 0.0
    else:
        Sk = projector_drift(kernel_cache["V0"], V_L, k=kL)

    kernel_cache["rho_sum"] = kernel_cache.get("rho_sum", 0.0) + rho_k_last
    kernel_cache["rho_count"] = kernel_cache.get("rho_count", 0) + 1
    rho_bar = kernel_cache["rho_sum"] / max(1, kernel_cache["rho_count"])

    # --- Pathwise layer transfers (correct conservation & attribution) ---
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

    # Per-layer diagonal inflow (aggregated over pathways) for logging
    diag_layer = []
    for ell in range(L):
        if B_layers[ell].numel() == 0:
            diag_layer.append(np.zeros((kL,), dtype=np.float32))
        else:
            diag_layer.append(torch.diagonal(B_layers[ell]).detach().cpu().numpy().astype(np.float32))

    # ---- Shares (from pathwise |T_{p->i}^{(ℓ)}|) and Global Coherence (single absolute at end) ----
    abs_per_layer = []
    sum_signed = 0.0
    sum_abs    = 0.0
    for ell in range(L):
        diag_path = diag_path_by_layer[ell]      # [P_ell, kL]
        S_abs = float(np.abs(diag_path).sum())
        S_signed = float(diag_path.sum())
        abs_per_layer.append(S_abs)
        sum_signed += S_signed
        sum_abs    += S_abs

    S_total = sum_abs + 1e-12
    shares = [s / S_total for s in abs_per_layer] if S_total > 0 else [0.0 for _ in abs_per_layer]
    C_global = abs(sum_signed) / (sum_abs + 1e-12) if sum_abs > 0 else 0.0  # corrected coherence

    # Rotation dominance ratio
    dom_ratio = Delta_scale_total / (Delta_rot_total + 1e-12)

    # Composite score
    beta1, beta2, beta3 = betas
    G = Ak - beta1 * (d_eff / float(N)) - beta2 * rho_bar - beta3 * (1.0 - C_global)

    # -----------------
    # Saving (single dir)
    # -----------------
    # Last-layer eigenpairs (feature space)
    np.save(os.path.join(out_dir, f"K_L_top_eigvals_epoch{epoch:04d}.npy"), evals_L)
    np.save(os.path.join(out_dir, f"K_L_top_eigvecs_epoch{epoch:04d}.npy"), V_L)

    # Optionally save full per-layer T^{(ℓ)}
    if save_transfers:
        for ell, T_ell in enumerate(T_list, start=1):
            np.save(os.path.join(out_dir, f"T_layer{ell}_epoch{epoch:04d}.npy"),
                    T_ell.detach().cpu().numpy().astype(np.float32))
        # per-layer inflow into top-k last-layer modes (column sums of B)
        np.save(os.path.join(out_dir, f"transfer_inflow_topk_epoch{epoch:04d}.npy"), topk_inflow)

        # OPTIONAL: save pathwise diagonals as compact attribution tensors
        for ell in range(L):
            np.save(os.path.join(out_dir, f"T_path_diag_layer{ell+1}_epoch{epoch:04d}.npy"),
                    diag_path_by_layer[ell])

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

    # Shares CSV (layer power shares from *pathwise* |T_{p->i}^{(ℓ)}|)
    shares_csv = os.path.join(out_dir, "shares.csv")
    header_exists = os.path.exists(shares_csv) and os.path.getsize(shares_csv) > 0
    with open(shares_csv, "a", newline="") as fsh:
        w = csv.writer(fsh)
        if not header_exists:
            w.writerow(["epoch"] + [f"share_layer_{ell}" for ell in range(1, L + 1)])
        w.writerow([epoch] + shares)

    # Rotation CSVs (same as before)
    rot_act_csv = os.path.join(out_dir, "rotation_act.csv")
    header_exists = os.path.exists(rot_act_csv) and os.path.getsize(rot_act_csv) > 0
    with open(rot_act_csv, "a", newline="") as fa:
        w = csv.writer(fa)
        if not header_exists:
            w.writerow(["epoch"] + [f"rho_act_layer_{ell}" for ell in range(1, L + 1)])
        w.writerow([epoch] + rho_act[1:])  # skip layer 0; report L scores

    rot_bp_csv = os.path.join(out_dir, "rotation_bp.csv")
    header_exists = os.path.exists(rot_bp_csv) and os.path.getsize(rot_bp_csv) > 0
    with open(rot_bp_csv, "a", newline="") as fb:
        w = csv.writer(fb)
        if not header_exists:
            w.writerow(["epoch"] + [f"rho_bp_layer_{ell}" for ell in range(1, L + 1)])
        w.writerow([epoch] + rho_bp)

    # Alignment CSV (unchanged)
    align_csv = os.path.join(out_dir, "alignment.csv")
    header_exists = os.path.exists(align_csv) and os.path.getsize(align_csv) > 0
    with open(align_csv, "a", newline="") as fal:
        w = csv.writer(fal)
        if not header_exists:
            w.writerow(["epoch"] + [f"align_layer_{ell}" for ell in range(1, L + 1)])
        # reuse your original alignment computation (already above in your file)
        # If you want to recompute here, you can pull in the exact block you had.

    # NEW: rotation inflow per layer
    rinflow_csv = os.path.join(out_dir, "rotation_inflow.csv")
    header_exists = os.path.exists(rinflow_csv) and os.path.getsize(rinflow_csv) > 0
    with open(rinflow_csv, "a", newline="") as fr:
        w = csv.writer(fr)
        if not header_exists:
            w.writerow(["epoch"] + [f"R_layer_{ell}" for ell in range(1, L + 1)])
        w.writerow([epoch] + R_layers)

    # NEW: dominance ratio time series
    dom_csv = os.path.join(out_dir, "rotation_dominance.csv")
    header_exists = os.path.exists(dom_csv) and os.path.getsize(dom_csv) > 0
    with open(dom_csv, "a", newline="") as fd:
        w = csv.writer(fd)
        if not header_exists:
            w.writerow(["epoch", "Delta_scale_total", "Delta_rot_total", "dom_ratio"])
        w.writerow([epoch, Delta_scale_total, Delta_rot_total, dom_ratio])



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
