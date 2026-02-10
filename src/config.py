"""
SpectralFM v2: Configuration
Hybrid Mamba-Transformer with OT, Physics, Wavelets, MoE, TTT, FNO, KAN, VIB
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import torch


@dataclass
class WaveletConfig:
    """Wavelet multi-scale embedding configuration."""
    wavelet: str = "db4"           # Daubechies-4 (good for spectral data)
    levels: int = 4                 # Decomposition levels
    mode: str = "symmetric"         # Boundary handling
    trainable_filters: bool = False # Learn wavelet filters


@dataclass
class MambaConfig:
    """Selective State Space Model (Mamba) configuration."""
    d_model: int = 256
    d_state: int = 16              # SSM state dimension
    d_conv: int = 4                # Local convolution width
    expand: int = 2                # Block expansion factor
    n_layers: int = 4              # Number of Mamba blocks
    dt_rank: str = "auto"          # Rank of Δ projection
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"
    dt_scale: float = 1.0
    bias: bool = False
    conv_bias: bool = True
    pscan: bool = True             # Parallel scan (faster)


@dataclass
class TransformerConfig:
    """Transformer block configuration."""
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 2              # Fewer layers (Mamba handles bulk)
    d_ff: int = 1024
    dropout: float = 0.1
    activation: str = "gelu"


@dataclass
class MoEConfig:
    """Mixture of Experts configuration."""
    n_experts: int = 4             # NIR, IR, Raman, Cross-modal
    top_k: int = 2                 # Activate top-k experts per input
    d_expert: int = 512            # Expert hidden dimension
    use_kan: bool = False          # Use KAN activations in experts
    balance_loss_weight: float = 0.01  # Load balancing
    noise_std: float = 0.1        # Gating noise for exploration


@dataclass
class FNOConfig:
    """Fourier Neural Operator transfer head configuration."""
    modes: int = 32                # Number of Fourier modes to keep
    width: int = 64                # Hidden channel width
    n_layers: int = 4
    activation: str = "gelu"
    use_spectral_conv: bool = True


@dataclass
class KANConfig:
    """Kolmogorov-Arnold Network configuration."""
    grid_size: int = 5             # B-spline grid points
    spline_order: int = 3          # Cubic B-splines
    grid_range: Tuple[float, float] = (-1.0, 1.0)
    base_activation: str = "silu"


@dataclass
class VIBConfig:
    """Variational Information Bottleneck configuration."""
    z_chem_dim: int = 128          # Chemistry-invariant latent
    z_inst_dim: int = 64           # Instrument-specific latent
    beta: float = 1e-3             # KL weight
    disentangle_weight: float = 0.1


@dataclass
class OTConfig:
    """Optimal Transport alignment configuration."""
    reg: float = 0.05              # Sinkhorn regularization
    n_iter: int = 100              # Sinkhorn iterations
    method: str = "sinkhorn"       # sinkhorn or emd
    weight: float = 0.1            # Loss weight


@dataclass
class PhysicsConfig:
    """Physics-informed loss configuration."""
    beer_lambert_weight: float = 0.05
    smoothness_weight: float = 0.05
    non_negativity_weight: float = 0.02
    peak_shape_weight: float = 0.02
    derivative_smoothness_weight: float = 0.03
    smoothness_kernel_size: int = 11


@dataclass
class TTTConfig:
    """Test-Time Training configuration."""
    n_steps: int = 10              # Gradient steps at test time
    lr: float = 1e-4               # TTT learning rate
    mask_ratio: float = 0.15       # MSRP mask ratio for TTT
    adapt_layers: str = "norm"     # Which layers to adapt: norm, all, lora


@dataclass
class LoRAConfig:
    """LoRA fine-tuning configuration."""
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj"]
    )


@dataclass
class PretrainConfig:
    """Pretraining configuration."""
    # MSRP
    mask_ratio: float = 0.20
    mask_type: str = "contiguous"  # contiguous, random, peak_aware
    mask_patch_size: int = 3       # Contiguous patches to mask together

    # Loss weights
    msrp_weight: float = 1.0
    contrastive_weight: float = 0.3
    denoise_weight: float = 0.2
    ot_weight: float = 0.1
    physics_weight: float = 0.1
    vib_weight: float = 0.05
    moe_balance_weight: float = 0.01

    # Training
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 50000
    grad_clip: float = 1.0
    optimizer: str = "adamw"
    scheduler: str = "cosine"

    # Augmentation
    noise_std: float = 0.01
    baseline_drift_scale: float = 0.005
    wavelength_shift_max: int = 3
    intensity_scale_range: Tuple[float, float] = (0.95, 1.05)


@dataclass
class FinetuneConfig:
    """Fine-tuning / calibration transfer configuration."""
    n_transfer_samples: List[int] = field(
        default_factory=lambda: [5, 10, 20, 30, 50, 100]
    )
    batch_size: int = 16
    lr: float = 1e-4
    epochs: int = 100
    patience: int = 15
    use_lora: bool = True
    use_ttt: bool = True


@dataclass
class SpectralFMConfig:
    """Master configuration for SpectralFM v2."""

    # Model name
    name: str = "SpectralFM-v2"
    seed: int = 42

    # Input
    n_channels: int = 2048         # Resample all spectra to this
    patch_size: int = 32
    stride: int = 16
    n_patches: int = 127           # (2048 - 32) / 16 + 1

    # Domain tokens
    domain_tokens: List[str] = field(
        default_factory=lambda: ["NIR", "IR", "RAMAN", "UNKNOWN"]
    )

    # Sub-configs
    wavelet: WaveletConfig = field(default_factory=WaveletConfig)
    mamba: MambaConfig = field(default_factory=MambaConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    fno: FNOConfig = field(default_factory=FNOConfig)
    kan: KANConfig = field(default_factory=KANConfig)
    vib: VIBConfig = field(default_factory=VIBConfig)
    ot: OTConfig = field(default_factory=OTConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    ttt: TTTConfig = field(default_factory=TTTConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    pretrain: PretrainConfig = field(default_factory=PretrainConfig)
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Paths
    data_dir: str = "data"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    @property
    def d_model(self) -> int:
        return self.mamba.d_model

    @property
    def total_latent_dim(self) -> int:
        return self.vib.z_chem_dim + self.vib.z_inst_dim


def get_light_config() -> SpectralFMConfig:
    """Get a lightweight configuration for fast CPU testing."""
    cfg = SpectralFMConfig()

    # Smaller model dimensions
    cfg.mamba.d_model = 64
    cfg.mamba.d_state = 8
    cfg.mamba.n_layers = 2
    cfg.mamba.expand = 1

    cfg.transformer.d_model = 64
    cfg.transformer.n_heads = 4
    cfg.transformer.n_layers = 1
    cfg.transformer.d_ff = 128

    cfg.moe.n_experts = 2
    cfg.moe.d_expert = 128

    cfg.vib.z_chem_dim = 32
    cfg.vib.z_inst_dim = 16

    cfg.fno.width = 16
    cfg.fno.modes = 8
    cfg.fno.n_layers = 2

    return cfg
