"""
SpectralFM v2: Mixture of Experts (MoE) + KAN Activations

Instrument-specific expert specialization with sparse top-k gating.
Each expert can optionally use Kolmogorov-Arnold Network activations
for interpretability.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import math


class BSplineBasis(nn.Module):
    """B-spline basis functions for KAN."""

    def __init__(self, grid_size: int = 5, spline_order: int = 3,
                 grid_range: Tuple[float, float] = (-1.0, 1.0)):
        super().__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Create grid points
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.linspace(
            grid_range[0] - spline_order * h,
            grid_range[1] + spline_order * h,
            grid_size + 2 * spline_order + 1
        )
        self.register_buffer('grid', grid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate B-spline basis at x.

        Args:
            x: (...) any shape
        Returns:
            bases: (..., n_basis) basis function values
        """
        x = x.unsqueeze(-1)  # (..., 1)
        grid = self.grid  # (n_grid,)

        # Cox-de Boor recursion (order 0)
        bases = ((x >= grid[:-1]) & (x < grid[1:])).float()

        # Recurse up to spline_order
        for k in range(1, self.spline_order + 1):
            left = (x - grid[:-(k + 1)]) / (grid[k:-1] - grid[:-(k + 1)] + 1e-8)
            right = (grid[k + 1:] - x) / (grid[k + 1:] - grid[1:-k] + 1e-8)
            bases = left * bases[:, :, :-1] + right * bases[:, :, 1:]

        return bases  # (..., grid_size + spline_order)


class KANLinear(nn.Module):
    """KAN layer: learnable activation functions on edges.

    Instead of fixed activations on nodes (MLP), KAN has learnable
    spline-based functions on edges. This provides:
    - Better accuracy with smaller networks
    - Interpretable learned functions (can recover physics!)
    """

    def __init__(self, in_features: int, out_features: int,
                 grid_size: int = 5, spline_order: int = 3,
                 base_activation: str = "silu"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        n_basis = grid_size + spline_order

        # Spline basis
        self.basis = BSplineBasis(grid_size, spline_order)

        # Learnable spline coefficients (the "activation functions on edges")
        self.spline_weight = nn.Parameter(
            torch.randn(out_features, in_features, n_basis) * 0.1
        )

        # Base linear (residual connection for stability)
        self.base_linear = nn.Linear(in_features, out_features)

        # Base activation
        self.base_act = nn.SiLU() if base_activation == "silu" else nn.GELU()

        # Scale
        self.scale = nn.Parameter(torch.ones(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., in_features)
        Returns:
            out: (..., out_features)
        """
        # Base path (standard linear + activation)
        base = self.base_linear(self.base_act(x))

        # Spline path
        # Normalize input to grid range
        x_norm = torch.tanh(x)  # Map to [-1, 1]
        bases = self.basis(x_norm)  # (..., in_features, n_basis)

        # Apply spline weights: contract over (in_features, n_basis)
        # spline_weight: (out_features, in_features, n_basis)
        # bases: (..., in_features, n_basis)
        spline_out = torch.einsum('...ib,oib->...o', bases, self.spline_weight)

        return self.scale * (base + spline_out)


class Expert(nn.Module):
    """Single expert network — FFN with optional KAN activations."""

    def __init__(self, d_model: int, d_hidden: int = 512,
                 use_kan: bool = False, dropout: float = 0.1):
        super().__init__()
        self.use_kan = use_kan

        if use_kan:
            self.net = nn.Sequential(
                KANLinear(d_model, d_hidden),
                nn.Dropout(dropout),
                KANLinear(d_hidden, d_model),
                nn.Dropout(dropout),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(d_model, d_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_hidden, d_model),
                nn.Dropout(dropout),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MixtureOfExperts(nn.Module):
    """Sparse Mixture of Experts layer.

    Routes each token to top-k experts based on a learned gating network.
    Different experts specialize in different instrument types/modalities.

    Load balancing loss ensures all experts are utilized.
    """

    def __init__(self, d_model: int = 256, n_experts: int = 4,
                 top_k: int = 2, d_expert: int = 512,
                 use_kan: bool = False, noise_std: float = 0.1,
                 dropout: float = 0.1):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.noise_std = noise_std

        # Gating network
        self.gate = nn.Linear(d_model, n_experts, bias=False)

        # Expert networks
        self.experts = nn.ModuleList([
            Expert(d_model, d_expert, use_kan, dropout)
            for _ in range(n_experts)
        ])

        # Layer norm
        self.norm = nn.LayerNorm(d_model)

        # Tracking for analysis
        self._expert_counts = None

    def _compute_gating(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute sparse gating weights with noise for exploration.

        Returns:
            gates: (B*L, n_experts) gating weights (sparse, top-k)
            indices: (B*L, top_k) selected expert indices
        """
        logits = self.gate(x)  # (B*L, n_experts)

        # Add noise during training for exploration
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise

        # Top-k selection
        top_k_logits, top_k_indices = logits.topk(self.top_k, dim=-1)
        top_k_gates = F.softmax(top_k_logits, dim=-1)

        # Create sparse gate tensor (match softmax output dtype for AMP compatibility)
        gates = torch.zeros_like(logits, dtype=top_k_gates.dtype)
        gates.scatter_(1, top_k_indices, top_k_gates)

        return gates, top_k_indices

    def _balance_loss(self, gates: torch.Tensor) -> torch.Tensor:
        """Load balancing loss to prevent expert collapse.

        Encourages uniform expert utilization across the batch.
        """
        # Fraction of tokens routed to each expert
        expert_load = gates.mean(dim=0)  # (n_experts,)
        # Target: uniform 1/n_experts
        target = torch.ones_like(expert_load) / self.n_experts
        return F.mse_loss(expert_load, target)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, L, d_model)
        Returns:
            output: (B, L, d_model)
            balance_loss: scalar
        """
        B, L, D = x.shape
        residual = x
        x_flat = self.norm(x).reshape(-1, D)  # (B*L, D)

        # Compute gating
        gates, indices = self._compute_gating(x_flat)  # (B*L, n_experts)

        # Store expert counts for analysis
        self._expert_counts = gates.sum(dim=0).detach()

        # Dispatch to experts
        output = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = gates[:, i] > 0  # Which tokens go to this expert
            if mask.any():
                expert_input = x_flat[mask]
                expert_output = expert(expert_input)
                output[mask] += gates[mask, i].unsqueeze(-1) * expert_output

        # Balance loss
        bal_loss = self._balance_loss(gates)

        output = output.reshape(B, L, D) + residual
        return output, bal_loss
