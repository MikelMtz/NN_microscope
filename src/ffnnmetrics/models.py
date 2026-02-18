from __future__ import annotations
import torch
import torch.nn as nn
from typing import List, Dict, Tuple

# Import torchvision for ResNet18 (NEW model)
import torchvision.models as models

class MLP(nn.Module):
    """
    Depth-L MLP with constant hidden width.
    Exposes forward caches (preacts a_l, activations h_l) and backprop errors δ_l.
    h0=x; a_l=W_l h_{l-1}+b_l; h_l=act(a_l); f=v^T h_L + b_out.
    """
    def __init__(self, d_in: int, width: int, depth: int, activation: str = "silu", num_classes: int = 10):
        super().__init__()
        assert depth >= 1, "depth must be >= 1"
        self.d_in, self.width, self.depth, self.num_classes = d_in, width, depth, num_classes
        self.act_name = activation.lower()
        self.act = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "tanh": nn.Tanh(),
        }.get(self.act_name, nn.SiLU())

        layers: List[nn.Module] = []
        in_dim = d_in
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, width, bias=True))
            in_dim = width
        self.hidden = nn.ModuleList(layers)
        self.out = nn.Linear(width, num_classes, bias=True)  # Updated to output logits for all classes

        for lin in self.hidden:
            if self.act_name in ("relu", "silu", "gelu"):
                nn.init.kaiming_normal_(lin.weight, nonlinearity="relu")
            else:
                nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)
        nn.init.xavier_normal_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_cache(x)["f"]

    def activation_derivative(self, a: torch.Tensor) -> torch.Tensor:
        if self.act_name == "relu":
            return (a > 0).to(a.dtype)
        if self.act_name == "silu":
            sig = torch.sigmoid(a)
            return sig + a * sig * (1.0 - sig)
        if self.act_name == "gelu":
            a = a.detach().requires_grad_(True)
            y = torch.nn.functional.gelu(a)
            (grad,) = torch.autograd.grad(y, a, torch.ones_like(y), retain_graph=False, create_graph=False)
            a.requires_grad_(False)
            return grad
        if self.act_name == "tanh":
            t = torch.tanh(a)
            return 1 - t * t
        a = a.detach().requires_grad_(True)
        y = self.act(a)
        (grad,) = torch.autograd.grad(y, a, torch.ones_like(y), retain_graph=False, create_graph=False)
        a.requires_grad_(False)
        return grad

    def forward_with_cache(self, x: torch.Tensor):
        cache = {"a": [], "h": []}
        h = x
        for lin in self.hidden:
            a = lin(h)
            cache["a"].append(a)
            h = self.act(a)
            cache["h"].append(h)
        f = self.out(h)
        cache["f"], cache["h0"] = f, x
        return cache

    @torch.no_grad()
    def get_layer_params(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        params = []
        for lin in self.hidden:
            params.append((lin.weight, lin.bias))
        params.append((self.out.weight, self.out.bias))
        return params

# New ResNet18 model for classification
class ResNet18(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

