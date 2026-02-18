# training/ntk_regularizers.py
import torch
import torch.nn.functional as F

def compute_last_layer_features(model, dataloader, device, max_samples=2048):
    """
    Collects penultimate-layer activations Phi for up to max_samples examples.
    Assumes model.forward returns (logits, features) OR model has hook to extract penultimate activations.
    Return: Phi [n_samples x p]
    """
    model.eval()
    feats = []
    seen = 0
    with torch.no_grad():
        for xb, _ in dataloader:
            xb = xb.to(device)
            logits, phi = model.forward_with_feature(xb)  # adapt to your model
            feats.append(phi.detach().cpu())
            seen += phi.size(0)
            if seen >= max_samples:
                break
    Phi = torch.cat(feats, dim=0)[:max_samples]  # [n x p]
    model.train()
    return Phi  # on CPU (float32)


def topk_eig(Phi, k):
    """
    Compute top-k eigenvalues and eigenvectors of empirical covariance C = (1/n) Phi^T Phi.
    Phi: [n x p] (torch.Tensor cpu)
    Returns: lambdas [k], U [p x k] (orthonormal, CPU tensors)
    """
    n, p = Phi.shape
    # small-p vs large-p strategy: use covariance in feature-space if p <= n; else use Nyström trick
    # We'll use p x p SVD if p is moderate (last-layer width).
    # C = (1/n) Phi^T Phi
    Phi = Phi.float()
    C = (Phi.t() @ Phi) / float(n)  # p x p
    # numeric safety: make symmetric
    C = (C + C.t()) / 2.0
    # use eigh
    vals, vecs = torch.linalg.eigh(C)  # ascending
    vals = vals.flip(0)
    vecs = vecs.flip(1)
    k = min(k, vals.shape[0])
    top_vals = vals[:k].clone()      # descending
    top_vecs = vecs[:, :k].clone()   # p x k
    return top_vals, top_vecs


def projection_matrix(U):
    # U: [p x k] orthonormal columns
    return U @ U.t()  # p x p


class NTKRegularizer:
    def __init__(self, k_top=32, gamma_eig=1e-2, gamma_sub=1e-2, device='cpu',
                 ref_mode='init', running_momentum=0.9):
        self.k = k_top
        self.gamma_eig = gamma_eig
        self.gamma_sub = gamma_sub
        self.device = device
        self.ref_mode = ref_mode
        self.momentum = running_momentum
        self.ref_eigs = None    # torch.Tensor [k]
        self.ref_subP = None    # torch.Tensor [p x p]  (or keep low-rank Uref)
        self.p = None

    def initialize_reference(self, Phi):  # call at start or first eval
        vals, U = topk_eig(Phi, self.k)
        self.p = U.shape[0]
        self.ref_eigs = vals.clone()
        # keep low-rank reference Uref to compute P_ref cheaply if needed
        self.U_ref = U.clone()
        self.ref_subP = projection_matrix(self.U_ref)  # on CPU

    def update_running_ref(self, Phi):
        vals, U = topk_eig(Phi, self.k)
        vals = vals.clone()
        U = U.clone()
        if self.ref_eigs is None:
            self.ref_eigs = vals
            self.U_ref = U
            self.ref_subP = projection_matrix(U)
            return
        # exponential moving average on eigenvalues (and on projection by mixing U in subspace sense)
        self.ref_eigs = self.momentum * self.ref_eigs + (1 - self.momentum) * vals
        # for subspace reference we update U_ref by aligning subspaces via SVD of (U_ref^T U)
        # simpler: keep projection matrix EMA
        P_new = projection_matrix(U)
        self.ref_subP = self.momentum * self.ref_subP + (1 - self.momentum) * P_new
        # optionally re-orthonormalize U_ref by top-k eigendecomp of ref_subP
        # For efficiency skip unless you want orthonormal U_ref.

    def eig_loss(self, lambdas):
        # lambdas: [k]
        # protect against zero ref
        ref = self.ref_eigs.to(lambdas.device)
        denom = torch.where(ref > 1e-12, ref, torch.ones_like(ref))
        rel = lambdas.to(ref.device) / denom - 1.0
        return (rel ** 2).sum() * self.gamma_eig

    def subspace_loss(self, U):
        # U: [p x k] current top eigenvectors
        P_t = projection_matrix(U.to(self.ref_subP.device))
        P_ref = self.ref_subP.to(P_t.device)
        diff = P_t - P_ref
        return (diff ** 2).sum() * self.gamma_sub

    def compute_losses_from_Phi(self, Phi):
        # Phi: [n x p] cpu tensor
        lambdas, U = topk_eig(Phi, self.k)
        # convert to training device
        lambdas = lambdas.to(self.device)
        U = U.to(self.device)
        L_eig = self.eig_loss(lambdas)
        L_sub = self.subspace_loss(U)
        return L_eig, L_sub, lambdas.detach().cpu(), U.detach().cpu()


##############################################
#   How to integrate in your training loop   #
##############################################

# in train.py (pseudo)
ntk = NTKRegularizer(k_top=config.ntk.k_top, gamma_eig=config.ntk.gamma_eig, gamma_sub=config.ntk.gamma_sub, device=device, ref_mode=config.ntk.ref_mode)

# Prepare small dataloader for feature estimation (can be a subset of trainset)
feature_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)

for step, (xb, yb) in enumerate(train_loader):
    # standard forward/backward
    logits = model(xb)
    loss = criterion(logits, yb)

    # every M steps compute/refresh Phi and add reg losses
    if step % config.ntk.eval_every_steps == 0:
        Phi = compute_last_layer_features(model, feature_loader, device, max_samples=1024)  # CPU tensor
        if ntk.ref_eigs is None:
            ntk.initialize_reference(Phi)  # set initial references
        elif config.ntk.ref_mode == 'running':
            ntk.update_running_ref(Phi)
        L_eig, L_sub, lambdas_cpu, U_cpu = ntk.compute_losses_from_Phi(Phi)
        loss = loss + L_eig + L_sub
        # optional: log lambdas_cpu for diagnostics

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
