"""
SpectralFM v2: Loss Functions

Complete loss suite:
- L_MSRP: Masked Spectrum Reconstruction (core pretraining)
- L_contrastive: BYOL-style instrument-invariance
- L_denoise: Denoising autoencoder
- L_OT: Optimal Transport alignment (Sinkhorn)
- L_physics: Beer-Lambert + smoothness + non-negativity + peak shape
- L_VIB: Variational Information Bottleneck
- L_MoE: Expert load balancing
- L_total: Weighted combination
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math


# ============================================================
# Core Pretraining Losses
# ============================================================

class MSRPLoss(nn.Module):
    """Masked Spectrum Reconstruction Pretraining loss.

    Only compute loss on masked patches (like BERT's MLM).
    """

    def __init__(self, norm: str = "l2"):
        super().__init__()
        self.norm = norm

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, N_patches, patch_size) predicted patch values
            target: (B, N_patches, patch_size) original patch values
            mask: (B, N_patches) binary mask (1 = masked)
        """
        # Only compute on masked patches
        mask_expanded = mask.unsqueeze(-1)  # (B, N, 1)

        if self.norm == "l2":
            loss = F.mse_loss(pred * mask_expanded, target * mask_expanded,
                              reduction='sum')
        elif self.norm == "l1":
            loss = F.l1_loss(pred * mask_expanded, target * mask_expanded,
                             reduction='sum')
        else:
            loss = F.smooth_l1_loss(pred * mask_expanded, target * mask_expanded,
                                    reduction='sum')

        # Normalize by number of masked elements
        n_masked = mask.sum() * pred.size(-1) + 1e-8
        return loss / n_masked


class ContrastiveLoss(nn.Module):
    """BYOL-style contrastive loss for instrument invariance.

    Same sample measured on different instruments should have similar
    representations, while different samples should have different ones.
    """

    def __init__(self, temperature: float = 0.1, projection_dim: int = 128):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """InfoNCE contrastive loss.

        Args:
            z1: (B, D) representations from instrument 1
            z2: (B, D) representations from instrument 2 (same samples)
        """
        # L2 normalize
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        # Similarity matrix
        B = z1.size(0)
        logits = torch.matmul(z1, z2.T) / self.temperature  # (B, B)

        # Positive pairs are on the diagonal
        labels = torch.arange(B, device=z1.device)
        loss = F.cross_entropy(logits, labels)

        return loss


class DenoisingLoss(nn.Module):
    """Denoising autoencoder loss.

    Input: noisy spectrum. Target: clean spectrum.
    Model learns to denoise, which builds robust representations.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor,
                clean_target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, clean_target)


# ============================================================
# Optimal Transport Alignment Loss
# ============================================================

class SinkhornDistance(nn.Module):
    """Sinkhorn divergence for optimal transport alignment.

    Aligns latent distributions from different instruments.
    Uses entropic regularization for differentiability.
    """

    def __init__(self, reg: float = 0.05, n_iter: int = 100, p: int = 2):
        super().__init__()
        self.reg = reg
        self.n_iter = n_iter
        self.p = p

    def _cost_matrix(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Pairwise cost matrix (Euclidean distance)."""
        n = x.size(0)
        m = y.size(0)
        x_sq = (x ** 2).sum(dim=1, keepdim=True)
        y_sq = (y ** 2).sum(dim=1, keepdim=True)
        cost = x_sq + y_sq.T - 2 * torch.matmul(x, y.T)
        return cost.clamp(min=0)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute Sinkhorn distance between two distributions.

        Args:
            x: (N, D) samples from source instrument distribution
            y: (M, D) samples from target instrument distribution
        Returns:
            distance: scalar Sinkhorn distance
        """
        n = x.size(0)
        m = y.size(0)

        # Uniform marginals
        mu = torch.ones(n, device=x.device) / n
        nu = torch.ones(m, device=x.device) / m

        # Cost matrix
        C = self._cost_matrix(x, y)

        # Sinkhorn iterations
        K = torch.exp(-C / self.reg)  # Gibbs kernel

        u = torch.ones(n, device=x.device)
        for _ in range(self.n_iter):
            v = nu / (K.T @ u + 1e-8)
            u = mu / (K @ v + 1e-8)

        # Transport plan
        pi = u.unsqueeze(1) * K * v.unsqueeze(0)

        # Wasserstein distance
        distance = (pi * C).sum()

        return distance


class OTAlignmentLoss(nn.Module):
    """Optimal Transport alignment loss for multi-instrument pretraining.

    Minimizes Wasserstein distance between latent distributions from
    different instruments, encouraging instrument-invariant representations.
    """

    def __init__(self, reg: float = 0.05, n_iter: int = 100):
        super().__init__()
        self.sinkhorn = SinkhornDistance(reg, n_iter)

    def forward(self, z_by_instrument: Dict[int, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            z_by_instrument: dict mapping instrument_id → (N_i, D) latent vectors
        Returns:
            loss: scalar OT alignment loss (average over all pairs)
        """
        inst_ids = list(z_by_instrument.keys())
        if len(inst_ids) < 2:
            return torch.tensor(0.0, device=list(z_by_instrument.values())[0].device)

        total_loss = 0.0
        n_pairs = 0

        for i in range(len(inst_ids)):
            for j in range(i + 1, len(inst_ids)):
                z_i = z_by_instrument[inst_ids[i]]
                z_j = z_by_instrument[inst_ids[j]]
                if len(z_i) > 0 and len(z_j) > 0:
                    total_loss += self.sinkhorn(z_i, z_j)
                    n_pairs += 1

        return total_loss / max(n_pairs, 1)


# ============================================================
# Physics-Informed Losses
# ============================================================

class PhysicsInformedLoss(nn.Module):
    """Physics-informed regularization for spectroscopy.

    Embeds known physical laws as soft constraints:
    1. Beer-Lambert: Absorbance should be linearly related to concentration
    2. Smoothness: Spectra should be smooth (not oscillating wildly)
    3. Non-negativity: Absorbance should be non-negative
    4. Peak shape: Peaks should follow Gaussian/Lorentzian profiles
    5. Derivative smoothness: Second derivative should be smooth
    """

    def __init__(self, smoothness_weight: float = 0.05,
                 non_neg_weight: float = 0.02,
                 derivative_weight: float = 0.03,
                 peak_shape_weight: float = 0.02,
                 kernel_size: int = 11):
        super().__init__()
        self.smoothness_weight = smoothness_weight
        self.non_neg_weight = non_neg_weight
        self.derivative_weight = derivative_weight
        self.peak_shape_weight = peak_shape_weight
        self.kernel_size = kernel_size

    def smoothness_loss(self, spectrum: torch.Tensor) -> torch.Tensor:
        """Penalize non-smooth spectra (high total variation)."""
        # First derivative magnitude (total variation)
        diff = spectrum[:, 1:] - spectrum[:, :-1]
        return (diff ** 2).mean()

    def non_negativity_loss(self, spectrum: torch.Tensor) -> torch.Tensor:
        """Penalize negative values in absorbance spectra."""
        return F.relu(-spectrum).mean()

    def derivative_smoothness_loss(self, spectrum: torch.Tensor) -> torch.Tensor:
        """Penalize rough second derivatives (Savitzky-Golay principle)."""
        # Second derivative
        d2 = spectrum[:, 2:] - 2 * spectrum[:, 1:-1] + spectrum[:, :-2]
        # Third derivative (smoothness of second derivative)
        d3 = d2[:, 1:] - d2[:, :-1]
        return (d3 ** 2).mean()

    def peak_symmetry_loss(self, spectrum: torch.Tensor) -> torch.Tensor:
        """Soft constraint: peaks should be approximately symmetric.

        Real peaks follow Gaussian/Lorentzian/Voigt profiles, all symmetric.
        """
        # Find local maxima (simplified: check neighbors)
        left = spectrum[:, :-2]
        center = spectrum[:, 1:-1]
        right = spectrum[:, 2:]

        is_peak = (center > left) & (center > right)
        # Compare left and right shoulders of peaks
        left_diff = center - left
        right_diff = center - right
        # Symmetric peaks: left_diff ≈ right_diff
        asymmetry = (left_diff - right_diff) ** 2
        # Only penalize at peak locations
        return (asymmetry * is_peak.float()).sum() / (is_peak.float().sum() + 1e-8)

    def forward(self, spectrum: torch.Tensor,
                reconstructed: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            spectrum: (B, L) original or reconstructed spectrum
            reconstructed: (B, L) if provided, compute physics losses on reconstruction

        Returns:
            dict with individual loss terms and total
        """
        target = reconstructed if reconstructed is not None else spectrum

        losses = {}
        losses["smoothness"] = self.smoothness_weight * self.smoothness_loss(target)
        losses["non_negativity"] = self.non_neg_weight * self.non_negativity_loss(target)
        losses["derivative"] = self.derivative_weight * self.derivative_smoothness_loss(target)
        losses["peak_shape"] = self.peak_shape_weight * self.peak_symmetry_loss(target)

        losses["total_physics"] = sum(losses.values())
        return losses


# ============================================================
# VIB Loss
# ============================================================

class VIBLoss(nn.Module):
    """Variational Information Bottleneck loss.

    Ensures z_chem captures chemistry (predicts targets, not instruments)
    and z_inst captures instrument info (predicts instruments, not targets).
    """

    def __init__(self, beta: float = 1e-3, disentangle_weight: float = 0.1):
        super().__init__()
        self.beta = beta
        self.disentangle_weight = disentangle_weight

    def forward(self, vib_out: Dict, instrument_id: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            vib_out: output from VIBHead
            instrument_id: (B,) ground truth instrument IDs
        """
        losses = {}

        # KL divergence (information bottleneck)
        losses["kl_chem"] = self.beta * vib_out["kl_chem"]
        losses["kl_inst"] = self.beta * vib_out["kl_inst"]

        # z_inst should predict instrument
        losses["inst_cls"] = F.cross_entropy(
            vib_out["inst_from_inst"], instrument_id
        ) * self.disentangle_weight

        # z_chem should NOT predict instrument (adversarial)
        # Minimize KL(uniform || pred) → push pred toward uniform → max entropy
        inst_pred_from_chem = F.softmax(vib_out["inst_from_chem"], dim=-1)
        n_inst = vib_out["inst_from_chem"].size(-1)
        uniform = torch.ones_like(inst_pred_from_chem) / n_inst
        losses["chem_adversarial"] = F.kl_div(
            inst_pred_from_chem.log(), uniform, reduction='batchmean'
        ) * self.disentangle_weight

        losses["total_vib"] = sum(losses.values())
        return losses


# ============================================================
# Combined Pretraining Loss
# ============================================================

class SpectralFMPretrainLoss(nn.Module):
    """Combined pretraining loss with all components."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        pc = config.pretrain

        self.msrp = MSRPLoss()
        self.contrastive = ContrastiveLoss()
        self.denoise = DenoisingLoss()
        self.ot = OTAlignmentLoss(config.ot.reg, config.ot.n_iter)
        self.physics = PhysicsInformedLoss(
            config.physics.smoothness_weight,
            config.physics.non_negativity_weight,
            config.physics.derivative_smoothness_weight,
            config.physics.peak_shape_weight,
            config.physics.smoothness_kernel_size,
        )
        self.vib_loss = VIBLoss(config.vib.beta, config.vib.disentangle_weight)

        # Weights
        self.w_msrp = pc.msrp_weight
        self.w_contrastive = pc.contrastive_weight
        self.w_denoise = pc.denoise_weight
        self.w_ot = pc.ot_weight
        self.w_physics = pc.physics_weight
        self.w_vib = pc.vib_weight
        self.w_moe = pc.moe_balance_weight

    def forward(self, model_output: Dict,
                instrument_id: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Compute all pretraining losses.

        Args:
            model_output: from SpectralFMForPretraining.forward()
            instrument_id: (B,) instrument IDs
        """
        losses = {}

        # 1. MSRP (core)
        losses["msrp"] = self.w_msrp * self.msrp(
            model_output["reconstruction"],
            model_output["target_patches"],
            model_output["mask"],
        )

        # 2. Physics
        # Reconstruct full spectrum from patches for physics loss
        physics_losses = self.physics(model_output["target_patches"].reshape(
            model_output["target_patches"].size(0), -1
        ))
        losses["physics"] = self.w_physics * physics_losses["total_physics"]

        # 3. VIB
        if instrument_id is not None:
            vib_losses = self.vib_loss(model_output["vib"], instrument_id)
            losses["vib"] = self.w_vib * vib_losses["total_vib"]
        else:
            losses["vib"] = torch.tensor(0.0, device=model_output["mask"].device)

        # 4. MoE balance
        losses["moe_balance"] = self.w_moe * model_output["moe_loss"]

        # 5. OT (computed separately when multi-instrument batch available)
        # Will be added in training loop

        # Total
        losses["total"] = sum(v for k, v in losses.items() if k != "total")

        return losses


class CalibrationTransferLoss(nn.Module):
    """Loss for fine-tuning calibration transfer."""

    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, 1) predicted property values
            target: (B,) ground truth
        """
        return F.mse_loss(pred.squeeze(-1), target)


# Alias for backward compatibility
PhysicsLoss = PhysicsInformedLoss
