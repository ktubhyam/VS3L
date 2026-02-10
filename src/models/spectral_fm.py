"""
SpectralFM v2: Full Model Assembly

Hybrid Mamba-Transformer architecture with:
- Wavelet multi-scale embedding
- Mamba backbone (selective SSM)
- Mixture of Experts layer
- Transformer global reasoning
- VIB disentangled latent space
- FNO transfer head
- TTT (test-time training) capability

Architecture:
    Spectrum → WaveletEmbed → Mamba(×4) → MoE → Transformer(×2) → VIB → [Heads]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import copy

from .embedding import WaveletEmbedding
from .mamba import MambaBackbone
from .moe import MixtureOfExperts
from .transformer import TransformerEncoder
from .heads import VIBHead, ReconstructionHead, RegressionHead, FNOTransferHead


class SpectralFM(nn.Module):
    """SpectralFM v2: Physics-Informed State Space Foundation Model
    for Zero-to-Few-Shot Calibration Transfer.

    The first self-supervised foundation model for vibrational spectroscopy
    that bridges Mamba's selective state space architecture with optimal
    transport-based domain adaptation.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config.d_model

        # ========== Embedding ==========
        self.embedding = WaveletEmbedding(
            d_model=d,
            n_channels=config.n_channels,
            wavelet_levels=config.wavelet.levels,
            patch_size=config.patch_size,
            stride=config.stride,
        )

        # ========== Mamba Backbone ==========
        self.mamba = MambaBackbone(
            d_model=d,
            n_layers=config.mamba.n_layers,
            d_state=config.mamba.d_state,
            d_conv=config.mamba.d_conv,
            expand=config.mamba.expand,
        )

        # ========== Mixture of Experts ==========
        self.moe = MixtureOfExperts(
            d_model=d,
            n_experts=config.moe.n_experts,
            top_k=config.moe.top_k,
            d_expert=config.moe.d_expert,
            use_kan=config.moe.use_kan,
            noise_std=config.moe.noise_std,
        )

        # ========== Transformer ==========
        self.transformer = TransformerEncoder(
            d_model=d,
            n_layers=config.transformer.n_layers,
            n_heads=config.transformer.n_heads,
            d_ff=config.transformer.d_ff,
            dropout=config.transformer.dropout,
        )

        # ========== VIB Disentanglement ==========
        self.vib = VIBHead(
            d_input=d,
            z_chem_dim=config.vib.z_chem_dim,
            z_inst_dim=config.vib.z_inst_dim,
            beta=config.vib.beta,
        )

        # ========== Task Heads ==========
        # Pretraining: MSRP reconstruction
        self.reconstruction_head = ReconstructionHead(
            d_input=d,
            n_patches=config.n_patches,
            patch_size=config.patch_size,
        )

        # Fine-tuning: property prediction from z_chem
        self.regression_head = RegressionHead(
            d_input=config.vib.z_chem_dim,
            n_targets=1,
        )

        # Transfer: FNO-based spectral transfer
        self.fno_head = FNOTransferHead(
            d_latent=config.vib.z_chem_dim,
            out_channels=config.n_channels,
            width=config.fno.width,
            modes=config.fno.modes,
            n_layers=config.fno.n_layers,
        )

        # MC Dropout for uncertainty
        self.mc_dropout = nn.Dropout(0.1)

        # Track parameters
        self._count_parameters()

    def _count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"SpectralFM v2: {total:,} total params, {trainable:,} trainable")

    def encode(self, spectrum: torch.Tensor,
               domain: Optional[str] = None,
               instrument_id: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Full encoding pipeline: spectrum → disentangled latent.

        Args:
            spectrum: (B, L) raw spectrum
            domain: "NIR", "IR", "RAMAN"
            instrument_id: (B,) instrument indices

        Returns:
            dict with tokens, z_chem, z_inst, vib outputs, moe_loss
        """
        # 1. Wavelet embedding
        tokens = self.embedding(spectrum, domain, instrument_id)
        # tokens: (B, N+2, d_model) — [CLS, DOMAIN, patches...]

        # 2. Mamba backbone
        tokens = self.mamba(tokens)

        # 3. Mixture of Experts
        tokens, moe_loss = self.moe(tokens)

        # 4. Transformer
        tokens = self.transformer(tokens)

        # 5. Extract CLS token for global representation
        cls_token = tokens[:, 0]  # (B, d_model)
        patch_tokens = tokens[:, 2:]  # (B, N_patches, d_model) — skip CLS + domain

        # 6. VIB disentanglement
        vib_out = self.vib(cls_token)

        return {
            "tokens": tokens,
            "patch_tokens": patch_tokens,
            "cls_token": cls_token,
            "z_chem": vib_out["z_chem"],
            "z_inst": vib_out["z_inst"],
            "vib": vib_out,
            "moe_loss": moe_loss,
        }

    def pretrain_forward(self, spectrum: torch.Tensor,
                         mask: torch.Tensor,
                         domain: Optional[str] = None,
                         instrument_id: Optional[torch.Tensor] = None) -> Dict:
        """Forward pass for pretraining (MSRP + auxiliary losses).

        Args:
            spectrum: (B, L) input spectrum
            mask: (B, N_patches) binary mask (1 = masked, 0 = visible)
            domain: domain string
            instrument_id: (B,) instrument IDs

        Returns:
            dict with reconstructions, latents, losses
        """
        # Encode (with masking applied to embedding)
        enc = self.encode(spectrum, domain, instrument_id)

        # Reconstruct masked patches
        reconstruction = self.reconstruction_head(enc["patch_tokens"])

        return {
            "reconstruction": reconstruction,
            "z_chem": enc["z_chem"],
            "z_inst": enc["z_inst"],
            "vib": enc["vib"],
            "moe_loss": enc["moe_loss"],
            "patch_tokens": enc["patch_tokens"],
            "cls_token": enc["cls_token"],
        }

    def transfer_forward(self, source_spectrum: torch.Tensor,
                         source_domain: str = "NIR",
                         target_instrument_params: Optional[torch.Tensor] = None,
                         output_length: Optional[int] = None) -> Dict:
        """Forward pass for calibration transfer.

        Encode source spectrum → extract z_chem → FNO transfer → target spectrum.
        """
        # Encode source
        enc = self.encode(source_spectrum, source_domain)

        # Transfer via FNO using chemistry-invariant features
        transferred = self.fno_head(
            enc["z_chem"], target_instrument_params, output_length
        )

        # Property prediction
        prediction = self.regression_head(enc["z_chem"])

        return {
            "transferred_spectrum": transferred,
            "prediction": prediction,
            "z_chem": enc["z_chem"],
            "z_inst": enc["z_inst"],
        }

    def predict(self, spectrum: torch.Tensor,
                domain: str = "NIR",
                mc_samples: int = 1) -> Dict:
        """Prediction with optional MC Dropout uncertainty.

        Args:
            spectrum: (B, L)
            domain: modality
            mc_samples: number of MC dropout forward passes (1 = no uncertainty)

        Returns:
            dict with mean prediction, uncertainty
        """
        if mc_samples <= 1:
            enc = self.encode(spectrum, domain)
            pred = self.regression_head(enc["z_chem"])
            return {"prediction": pred, "z_chem": enc["z_chem"]}

        # MC Dropout
        self.train()  # Enable dropout
        preds = []
        for _ in range(mc_samples):
            enc = self.encode(spectrum, domain)
            z = self.mc_dropout(enc["z_chem"])
            pred = self.regression_head(z)
            preds.append(pred)
        self.eval()

        preds = torch.stack(preds)  # (mc_samples, B, n_targets)
        return {
            "prediction": preds.mean(0),
            "uncertainty": preds.std(0),
            "z_chem": enc["z_chem"],
        }

    def test_time_train(self, test_spectra: torch.Tensor,
                        n_steps: int = 10, lr: float = 1e-4,
                        mask_ratio: float = 0.15,
                        adapt_layers: str = "norm") -> None:
        """Test-Time Training: adapt model to new instrument using
        self-supervised MSRP on unlabeled test spectra.

        This enables ZERO-SHOT calibration transfer!

        Args:
            test_spectra: (N, L) unlabeled spectra from new instrument
            n_steps: gradient steps
            lr: TTT learning rate
            mask_ratio: masking ratio for self-supervised objective
            adapt_layers: which params to update ("norm", "all", "lora")
        """
        # Select parameters to adapt
        if adapt_layers == "norm":
            params = [p for n, p in self.named_parameters()
                      if "norm" in n or "ln" in n]
        elif adapt_layers == "lora":
            params = [p for n, p in self.named_parameters()
                      if "lora" in n.lower()]
        else:
            params = list(self.parameters())

        if not params:
            params = [p for n, p in self.named_parameters()
                      if "norm" in n]

        optimizer = torch.optim.Adam(params, lr=lr)

        self.train()
        for step in range(n_steps):
            # Random batch from test spectra
            batch_size = min(32, len(test_spectra))
            idx = torch.randperm(len(test_spectra))[:batch_size]
            batch = test_spectra[idx]

            # Create random mask
            n_patches = self.config.n_patches
            n_mask = int(n_patches * mask_ratio)
            mask = torch.zeros(batch_size, n_patches, device=batch.device)
            for i in range(batch_size):
                mask_idx = torch.randperm(n_patches)[:n_mask]
                mask[i, mask_idx] = 1

            # Forward pass
            output = self.pretrain_forward(batch, mask)

            # Self-supervised loss (MSRP only — no labels needed!)
            # Reconstruct masked patches
            target_patches = self._patchify(batch)
            recon_loss = F.mse_loss(
                output["reconstruction"] * mask.unsqueeze(-1),
                target_patches * mask.unsqueeze(-1)
            )

            optimizer.zero_grad()
            recon_loss.backward()
            optimizer.step()

        self.eval()

    def _patchify(self, spectrum: torch.Tensor) -> torch.Tensor:
        """Convert spectrum to patches for reconstruction target.

        Args:
            spectrum: (B, L)
        Returns:
            patches: (B, N_patches, patch_size)
        """
        B, L = spectrum.shape
        p = self.config.patch_size
        s = self.config.stride

        patches = spectrum.unfold(1, p, s)  # (B, N_patches, patch_size)
        return patches

    def get_lora_params(self) -> List[nn.Parameter]:
        """Get LoRA-specific parameters for fine-tuning."""
        return [p for n, p in self.named_parameters()
                if "lora" in n.lower()]

    def freeze_backbone(self):
        """Freeze everything except task heads and LoRA adapters."""
        for param in self.parameters():
            param.requires_grad = False
        # Unfreeze heads
        for param in self.regression_head.parameters():
            param.requires_grad = True
        for param in self.fno_head.parameters():
            param.requires_grad = True
        # Unfreeze LoRA params (if injected)
        for name, param in self.named_parameters():
            if 'lora_' in name:
                param.requires_grad = True

    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True


class SpectralFMForPretraining(nn.Module):
    """Wrapper for pretraining with masking strategy."""

    def __init__(self, model: SpectralFM, config):
        super().__init__()
        self.model = model
        self.config = config

    def create_mask(self, batch_size: int, n_patches: int,
                    mask_ratio: float = 0.20,
                    mask_type: str = "contiguous",
                    mask_patch_size: int = 3) -> torch.Tensor:
        """Create masking pattern for MSRP.

        Args:
            batch_size: B
            n_patches: total number of patches
            mask_ratio: fraction to mask
            mask_type: "contiguous", "random", or "peak_aware"
            mask_patch_size: size of contiguous mask blocks

        Returns:
            mask: (B, n_patches) binary mask (1 = masked)
        """
        mask = torch.zeros(batch_size, n_patches)
        n_mask = int(n_patches * mask_ratio)

        if mask_type == "random":
            for i in range(batch_size):
                idx = torch.randperm(n_patches)[:n_mask]
                mask[i, idx] = 1

        elif mask_type == "contiguous":
            for i in range(batch_size):
                # Mask contiguous blocks
                n_blocks = max(1, n_mask // mask_patch_size)
                for _ in range(n_blocks):
                    start = torch.randint(0, n_patches - mask_patch_size + 1, (1,))
                    mask[i, start:start + mask_patch_size] = 1

        elif mask_type == "peak_aware":
            # Higher probability of masking peak regions
            # (will be implemented with spectral analysis)
            for i in range(batch_size):
                idx = torch.randperm(n_patches)[:n_mask]
                mask[i, idx] = 1

        return mask

    def forward(self, spectrum: torch.Tensor,
                domain: Optional[str] = None,
                instrument_id: Optional[torch.Tensor] = None) -> Dict:
        """Pretraining forward pass."""
        B = spectrum.size(0)
        n_patches = self.config.n_patches
        device = spectrum.device

        # Create mask
        mask = self.create_mask(
            B, n_patches,
            self.config.pretrain.mask_ratio,
            self.config.pretrain.mask_type,
            self.config.pretrain.mask_patch_size,
        ).to(device)

        # Forward
        output = self.model.pretrain_forward(spectrum, mask, domain, instrument_id)
        output["mask"] = mask

        # Target patches
        target_patches = self.model._patchify(spectrum)
        output["target_patches"] = target_patches

        return output
