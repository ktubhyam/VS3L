"""
LoRA (Low-Rank Adaptation) for SpectralFM fine-tuning.

Wraps existing linear layers with low-rank A*B decomposition.
During fine-tuning, only A and B are trained (rank * d parameters
instead of d * d), enabling efficient adaptation with ~0.5% of
backbone parameters.

Reference: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", 2021
"""
import torch
import torch.nn as nn
from typing import List
import math
import logging

logger = logging.getLogger(__name__)


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation.

    Forward: y = W_frozen @ x + (alpha/rank) * B @ A @ x

    W_frozen: original pretrained weights (frozen)
    A: (rank, in_features) — initialized with Kaiming uniform
    B: (out_features, rank) — initialized with zeros

    At init: B @ A = 0, so output = original W @ x (no change).
    """

    def __init__(self, original_linear: nn.Linear, rank: int = 8,
                 alpha: float = 16.0, dropout: float = 0.05):
        super().__init__()
        self.original = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # Freeze original weights
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Initialize A with Kaiming uniform (so A has signal, B starts at 0)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original forward (frozen)
        result = self.original(x)
        # LoRA adaptation
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        return result + lora_out

    def merge_weights(self):
        """Merge LoRA weights into original for inference (no overhead)."""
        with torch.no_grad():
            self.original.weight += self.scaling * (self.lora_B @ self.lora_A)

    @property
    def lora_parameters(self) -> List[nn.Parameter]:
        return [self.lora_A, self.lora_B]


def inject_lora(model: nn.Module, target_modules: List[str],
                rank: int = 8, alpha: float = 16.0,
                dropout: float = 0.05) -> nn.Module:
    """Inject LoRA adapters into specified modules of a model.

    Walks the model tree. For each module whose name ends with a string
    in target_modules, replaces the nn.Linear with a LoRALinear.

    Args:
        model: the pretrained SpectralFM model
        target_modules: list of substrings to match (e.g., ["q_proj", "k_proj", "v_proj"])
        rank: LoRA rank
        alpha: LoRA scaling factor
        dropout: LoRA dropout

    Returns:
        model with LoRA injected (same object, modified in-place)
    """
    lora_count = 0

    # Build name→module map once
    module_map = dict(model.named_modules())

    for name, module in list(model.named_modules()):
        for target in target_modules:
            if name.endswith(target) and isinstance(module, nn.Linear):
                # Get parent module and attribute name
                parts = name.rsplit('.', 1)
                if len(parts) == 2:
                    parent_name, attr_name = parts
                    parent = module_map[parent_name]
                else:
                    parent = model
                    attr_name = name

                # Replace with LoRA version
                lora_layer = LoRALinear(module, rank, alpha, dropout)
                setattr(parent, attr_name, lora_layer)
                lora_count += 1

    logger.info(f"Injected LoRA into {lora_count} layers (rank={rank}, alpha={alpha})")

    # Count LoRA params
    lora_params = sum(p.numel() for n, p in model.named_parameters() if 'lora_' in n)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"LoRA params: {lora_params:,} ({100*lora_params/total_params:.2f}% of total)")
    logger.info(f"Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}% of total)")

    return model


def get_lora_state_dict(model: nn.Module) -> dict:
    """Extract only LoRA parameters for lightweight saving."""
    return {k: v for k, v in model.state_dict().items() if 'lora_' in k}


def get_lora_optimizer_params(model: nn.Module, lr: float = 1e-4,
                               head_lr: float = 1e-3) -> List[dict]:
    """Get parameter groups for LoRA fine-tuning.

    Returns separate groups for:
    1. LoRA parameters (backbone adaptation) — lower LR
    2. Head parameters (task-specific) — higher LR
    """
    lora_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'lora_' in name:
            lora_params.append(param)
        elif 'regression_head' in name or 'fno_head' in name:
            head_params.append(param)

    return [
        {"params": lora_params, "lr": lr, "weight_decay": 0.01},
        {"params": head_params, "lr": head_lr, "weight_decay": 0.01},
    ]


def save_lora_weights(model: nn.Module, path: str) -> None:
    """Save only LoRA + head weights to a file."""
    state = {}
    for name, param in model.named_parameters():
        if 'lora_' in name or 'regression_head' in name or 'fno_head' in name:
            state[name] = param.data.cpu()
    torch.save(state, path)


def load_lora_weights(model: nn.Module, path: str) -> None:
    """Load LoRA + head weights from a file."""
    state = torch.load(path, map_location='cpu', weights_only=True)
    model_state = model.state_dict()
    for name, param in state.items():
        if name in model_state:
            model_state[name].copy_(param)
    model.load_state_dict(model_state)
