# Complete List of Implemented Metrics

This document provides a comprehensive list of all metrics computed by the NN_microscope framework, organized by category.

## 1. Basic Kernel Metrics (Last Layer)

These metrics are computed on the last layer's feature Gram matrix C^{(L)} and saved to `kernel/summary.csv`.

### A_k (Alignment)
- **Definition**: Cumulative alignment of the model output with the top-k eigenfunctions of the last-layer kernel.
- **Formula**: A_k = ||V_k^T H^T y||^2 / ||y||^2, where V_k are top-k eigenvectors and H is the last-layer features.
- **Interpretation**: Measures how well the model output aligns with the dominant modes of the feature kernel. Higher values (closer to 1) indicate better alignment.

### d_eff (Effective Dimension)
- **Definition**: Effective dimensionality of the feature space.
- **Formula**: d_eff = tr(C^2) / tr(C)^2
- **Interpretation**: Measures the effective rank of the feature covariance. Lower values indicate more concentrated features.

### DeltaC (Kernel Speed)
- **Definition**: Normalized rate of change of the last-layer Gram matrix.
- **Formula**: ||C_dot|| / ||C||
- **Interpretation**: Measures how fast the kernel is evolving during training.

### rho_k (Rotation Speed)
- **Definition**: Rotation speed of the top-k eigenspace of the last layer.
- **Formula**: sqrt(Σ_{i≠j} |M_{ji}|^2 / gap_{ij}^2), where M = V^T C_dot V
- **Interpretation**: Measures how fast the eigenvectors are rotating. Higher values indicate more rotation.

### rho_bar (Average Rotation Speed)
- **Definition**: Time-averaged rotation speed across epochs.
- **Interpretation**: Smoothed measure of rotation dynamics.

### S_k (Projector Drift)
- **Definition**: Drift of the projector onto the top-k eigenspace from initialization.
- **Formula**: ||P_t - P_0||_F, where P = V V^T
- **Interpretation**: Measures how much the dominant eigenspace has shifted from initialization.

### C (Global Coherence)
- **Definition**: Global coherence of pathwise transfers.
- **Formula**: |Σ T| / Σ |T|
- **Interpretation**: Measures how aligned the signed transfers are. Values close to 1 indicate coherent transfers.

### G (Composite Score)
- **Definition**: Composite metric combining alignment, effective dimension, rotation, and coherence.
- **Formula**: G = A_k - β₁(d_eff/N) - β₂ρ_bar - β₃(1-C)
- **Interpretation**: Overall quality metric (higher is better). β₁, β₂, β₃ are configurable weights.

### dom_ratio (Dominance Ratio)
- **Definition**: Ratio of scale changes to rotation changes.
- **Formula**: Δ_scale / Δ_rot
- **Interpretation**: Values > 1 indicate scale-dominant dynamics; < 1 indicates rotation-dominant.

### eigval_i (Eigenvalues)
- **Definition**: Top-k eigenvalues of the last-layer Gram matrix C^{(L)}.
- **Interpretation**: Spectrum of the feature kernel, ordered by magnitude.

---

## 2. Per-Layer Metrics

### share_layer_ℓ (Layer Share)
- **Definition**: Fraction of total pathwise transfer magnitude contributed by layer ℓ.
- **Formula**: |T^{(ℓ)}| / Σ_{ℓ'} |T^{(ℓ')}|
- **Saved to**: `kernel/shares.csv`
- **Interpretation**: Measures the relative contribution of each layer to feature evolution.

### rho_act_layer_ℓ (Activation Rotation Speed)
- **Definition**: Rotation speed of the eigenspace for activation features at layer ℓ.
- **Saved to**: `kernel/rotation_act.csv`
- **Interpretation**: Per-layer rotation dynamics in forward pass.

### rho_bp_layer_ℓ (Backprop Rotation Speed)
- **Definition**: Rotation speed of the eigenspace for backprop features at layer ℓ.
- **Saved to**: `kernel/rotation_bp.csv`
- **Interpretation**: Per-layer rotation dynamics in backward pass.

### align_layer_ℓ (Alignment)
- **Definition**: Alignment of layer ℓ's eigenspace with the last layer's eigenspace.
- **Formula**: A(ℓ→L) = (1/k_ℓ) ||V_L^T Sbar^{ℓ→L} V_ℓ||_F^2
- **Saved to**: `kernel/alignment.csv`
- **Interpretation**: Measures how well layer ℓ's dominant modes align with the final layer after transport.

---

## 3. Last-Layer Advanced Metrics

These metrics are computed on the last layer and saved to `kernel/lastlayer_advanced.csv`.

### LARD (Label-Aware Rotation Dominance)
- **Definition**: Ratio of label-weighted rotation to scale changes.
- **Formula**: (Σ_ℓ Σ_{i≠j} |B_{ji}^{(ℓ)}| |α_j α_i| / gap_{ij}) / (Σ_ℓ Σ_i |B_{ii}^{(ℓ)}| α_i^2)
- **Interpretation**: Values > 1 indicate rotation-dominant dynamics that are important for the labels. Measures how much rotation contributes relative to scaling.

### GSI (Gap-Stress Index)
- **Definition**: Total gap-stress from off-diagonal transfers.
- **Formula**: Σ_ℓ Σ_{i≠j} |B_{ji}^{(ℓ)}| / gap_{ij}^2
- **Interpretation**: Measures stress on the eigenspace from rotation. Higher values indicate more stress from mode mixing.

### RPE (Relative Progress Efficiency)
- **Definition**: Progress in alignment relative to rotation cost.
- **Formula**: ΔA_k / R_y, where R_y is the label-weighted rotation inflow.
- **Interpretation**: Measures how efficiently alignment improves relative to rotation. Higher values indicate efficient learning.

### H_path (Pathway Entropy)
- **Definition**: Shannon entropy over all pathway transfers |T_{p→i}^{(ℓ)}|.
- **Formula**: -Σ p log(p), where p is normalized transfer magnitudes
- **Interpretation**: Measures diversity of pathways. Higher entropy = more distributed pathways.

### Top5_mass (Top-5% Mass)
- **Definition**: Fraction of total transfer magnitude in the top 5% of pathways.
- **Interpretation**: Measures concentration of transfers. Higher values indicate more concentrated pathways.

### Gini (Gini Coefficient)
- **Definition**: Inequality measure of pathway transfers.
- **Formula**: Standard Gini coefficient over normalized transfer magnitudes
- **Interpretation**: Measures inequality in pathway usage. 0 = uniform, 1 = maximally concentrated.

### STR (Spectral Turnover Rate)
- **Definition**: Normalized number of pairwise order flips in top-k eigenvalues.
- **Formula**: (number of inversions in eigenvalue ranking) / (k(k-1)/2)
- **Interpretation**: Measures how much the eigenvalue ordering changes. Higher values indicate more spectral reordering.

### STB_i^{(ℓ)} (Signed Transfer Balance)
- **Definition**: Per-mode signed transfer balance for layer ℓ.
- **Formula**: STB_i = (Σ_p T_{p→i}) / (Σ_p |T_{p→i}|)
- **Saved to**: `kernel/STB_layer{ell}_epoch{epoch}.npy`
- **Interpretation**: Values in [-1, 1]. +1 means all transfers are positive, -1 means all negative, 0 means balanced.

---

## 4. Inter-Layer Metrics

These metrics measure transfers between any source layer ℓ and target layer r, saved to `kernel/interlayer_ilard.csv`.

### ILARD(ℓ→r) (Inter-Layer Label-Aware Rotation Dominance)
- **Definition**: Label-aware rotation dominance for transfers from layer ℓ to layer r.
- **Formula**: Same as LARD but computed on B^{(ℓ→r)} with target layer r's eigenstructure.
- **Interpretation**: Measures rotation vs scale dominance for inter-layer transfers.

### R_inflow(ℓ→r) (Rotation Inflow)
- **Definition**: Rotation inflow from layer ℓ to layer r.
- **Formula**: Σ_{i≠j} |B_{ji}^{(ℓ→r)}| / gap_{ij}
- **Interpretation**: Magnitude of rotation transfers between layers.

### I_GSI(ℓ→r) (Inter-Layer Gap-Stress Index)
- **Definition**: Gap-stress index for transfers from layer ℓ to layer r.
- **Formula**: Σ_{i≠j} |B_{ji}^{(ℓ→r)}| / gap_{ij}^2
- **Interpretation**: Stress on target layer r's eigenspace from source layer ℓ.

### H_path_pair(ℓ→r) (Inter-Layer Pathway Entropy)
- **Definition**: Shannon entropy of pathway transfers from layer ℓ to layer r.
- **Interpretation**: Diversity of pathways between specific layer pairs.

### Coherence_pair(ℓ→r) (Inter-Layer Coherence)
- **Definition**: Coherence of transfers from layer ℓ to layer r.
- **Formula**: |Σ T| / Σ |T| for transfers from ℓ to r
- **Interpretation**: How aligned the signed transfers are between layer pairs.

---

## 5. Depth Profile Metrics

Saved to `kernel/interlayer_depth_profiles.csv`.

### D_scale (Scale Depth)
- **Definition**: Weighted average depth of scale transfers.
- **Formula**: (Σ_{ℓ,r} |diag(B^{(ℓ→r)})| × (r-ℓ)) / (Σ_{ℓ,r} |diag(B^{(ℓ→r)})|)
- **Interpretation**: Average distance over which scale changes propagate.

### D_rot (Rotation Depth)
- **Definition**: Weighted average depth of rotation transfers.
- **Formula**: (Σ_{ℓ,r} (|off(B^{(ℓ→r)})|/gap) × (r-ℓ)) / (Σ_{ℓ,r} |off(B^{(ℓ→r)})|/gap)
- **Interpretation**: Average distance over which rotations propagate.

---

## 6. Transported Alignment

Saved to `kernel/transported_alignment.csv`.

### A_transport(ℓ→r)
- **Definition**: Alignment of layer ℓ's eigenspace with layer r's eigenspace after transport.
- **Formula**: A(ℓ→r) = (1/k_ℓ) ||V_r^T Sbar^{ℓ→r} V_ℓ||_F^2
- **Interpretation**: Measures how well source layer modes align with target layer modes after propagating through intermediate layers.

---

## 7. Transfer Inflow Metrics

Saved to `kernel/transfer_inflow_topk_epoch{epoch}.npy`.

### Transfer Inflow to Top-k Modes
- **Definition**: Column sums of B^{(ℓ)} matrices, showing total inflow into each top-k mode from layer ℓ.
- **Interpretation**: Which modes receive the most transfer from each layer.

---

## 8. Eigenvalue Metrics

Saved to `kernel/C_act_eigvals_layer{ell}_epoch{epoch}.npy` and `kernel/C_bp_eigvals_layer{ell}_epoch{epoch}.npy`.

### Activation Eigenvalues
- **Definition**: Top-k eigenvalues of activation feature Gram matrices C^{(ℓ)} for each layer ℓ.

### Backprop Eigenvalues
- **Definition**: Top-k eigenvalues of backprop feature Gram matrices for each layer ℓ.

---

## 9. Training Metrics

Saved to `metrics.csv`.

### train_mse
- **Definition**: Mean squared error on training set.

### test_mse
- **Definition**: Mean squared error on test set.

---

## Summary by File

- **`kernel/summary.csv`**: A_k, d_eff, DeltaC, rho_k, rho_bar, S_k, C, G, dom_ratio, eigval_i
- **`kernel/shares.csv`**: share_layer_ℓ for each layer
- **`kernel/rotation_act.csv`**: rho_act_layer_ℓ for each layer
- **`kernel/rotation_bp.csv`**: rho_bp_layer_ℓ for each layer
- **`kernel/alignment.csv`**: align_layer_ℓ for each layer
- **`kernel/lastlayer_advanced.csv`**: LARD, GSI, RPE, H_path, Top5_mass, Gini, STR
- **`kernel/interlayer_ilard.csv`**: ILARD, R_inflow, I_GSI, H_path_pair, Coherence_pair (for each ℓ→r pair)
- **`kernel/interlayer_depth_profiles.csv`**: D_scale, D_rot
- **`kernel/transported_alignment.csv`**: A_transport(ℓ→r) for each pair
- **`kernel/STB_layer{ell}_epoch{epoch}.npy`**: STB values per mode per layer
- **`kernel/transfer_inflow_topk_epoch{epoch}.npy`**: Transfer inflows per layer per mode
- **`kernel/C_act_eigvals_layer{ell}_epoch{epoch}.npy`**: Activation eigenvalues per layer
- **`kernel/C_bp_eigvals_layer{ell}_epoch{epoch}.npy`**: Backprop eigenvalues per layer
- **`metrics.csv`**: train_mse, test_mse


