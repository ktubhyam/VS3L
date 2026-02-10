#!/usr/bin/env python3
"""
SpectralFM v2 — Smoke Test Suite

Run: python tests/smoke_test.py
Tests each module independently, then tests the full forward/backward pass.
Fix issues module by module until all tests pass.
"""
import sys
import os
import traceback

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

PASS = "✅"
FAIL = "❌"
SKIP = "⏭️"

results = []

def test(name, fn):
    """Run a test function and record result."""
    try:
        fn()
        results.append((name, PASS, ""))
        print(f"  {PASS} {name}")
    except Exception as e:
        results.append((name, FAIL, str(e)))
        print(f"  {FAIL} {name}: {e}")
        traceback.print_exc()
        print()


def test_imports():
    """Test that all modules import correctly."""
    import torch
    print(f"  PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")

def test_config():
    from src.config import SpectralFMConfig
    cfg = SpectralFMConfig()
    assert cfg.d_model == 256
    assert cfg.n_channels == 2048
    assert cfg.mamba.n_layers == 4
    assert cfg.transformer.n_layers == 2

def test_data_corn():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                            "data", "processed", "corn")
    m5 = np.load(os.path.join(data_dir, "m5_spectra.npy"))
    mp6 = np.load(os.path.join(data_dir, "mp6_spectra.npy"))
    props = np.load(os.path.join(data_dir, "properties.npy"))
    wl = np.load(os.path.join(data_dir, "wavelengths.npy"))
    assert m5.shape == (80, 700), f"m5 shape: {m5.shape}"
    assert mp6.shape == (80, 700), f"mp6 shape: {mp6.shape}"
    assert props.shape == (80, 4), f"props shape: {props.shape}"
    assert wl.shape == (700,), f"wavelengths shape: {wl.shape}"

def test_data_tablet():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                            "data", "processed", "tablet")
    cal1 = np.load(os.path.join(data_dir, "calibrate_1.npy"))
    cal2 = np.load(os.path.join(data_dir, "calibrate_2.npy"))
    calY = np.load(os.path.join(data_dir, "calibrate_Y.npy"))
    assert cal1.shape == (155, 650), f"cal1 shape: {cal1.shape}"
    assert cal2.shape == (155, 650), f"cal2 shape: {cal2.shape}"
    assert calY.shape == (155, 3), f"calY shape: {calY.shape}"

def test_wavelet_embedding():
    import torch
    from src.config import SpectralFMConfig
    from src.models.embedding import WaveletEmbedding
    
    cfg = SpectralFMConfig()
    embed = WaveletEmbedding(
        d_model=cfg.d_model,
        n_channels=cfg.n_channels,
        wavelet_levels=cfg.wavelet.levels,
        patch_size=cfg.patch_size,
        stride=cfg.stride,
    )
    x = torch.randn(2, 2048)
    tokens = embed(x, domain="NIR")
    print(f"    WaveletEmbedding output: {tokens.shape}")
    # Should be (2, N+2, 256) where N = num patches
    assert tokens.dim() == 3
    assert tokens.shape[0] == 2
    assert tokens.shape[2] == cfg.d_model
    actual_n_patches = tokens.shape[1] - 2  # minus CLS + domain
    print(f"    Actual n_patches: {actual_n_patches} (config says {cfg.n_patches})")

def test_mamba():
    import torch
    from src.models.mamba import MambaBackbone
    
    mamba = MambaBackbone(d_model=256, n_layers=2, d_state=16, d_conv=4, expand=2)
    x = torch.randn(2, 64, 256)  # (B, L, D)
    y = mamba(x)
    assert y.shape == x.shape, f"Mamba output {y.shape} != input {x.shape}"

def test_moe():
    import torch
    from src.models.moe import MixtureOfExperts
    
    moe = MixtureOfExperts(d_model=256, n_experts=4, top_k=2, d_expert=512)
    x = torch.randn(2, 64, 256)
    y, balance_loss = moe(x)
    assert y.shape == x.shape, f"MoE output {y.shape} != input {x.shape}"
    assert balance_loss.dim() == 0, "Balance loss should be scalar"

def test_transformer():
    import torch
    from src.models.transformer import TransformerEncoder
    
    tf = TransformerEncoder(d_model=256, n_layers=2, n_heads=8, d_ff=1024)
    x = torch.randn(2, 64, 256)
    y = tf(x)
    assert y.shape == x.shape, f"Transformer output {y.shape} != input {x.shape}"

def test_vib_head():
    import torch
    from src.models.heads import VIBHead
    
    vib = VIBHead(d_input=256, z_chem_dim=128, z_inst_dim=64)
    x = torch.randn(2, 256)
    out = vib(x)
    assert out["z_chem"].shape == (2, 128), f"z_chem: {out['z_chem'].shape}"
    assert out["z_inst"].shape == (2, 64), f"z_inst: {out['z_inst'].shape}"
    assert "kl_loss" in out

def test_reconstruction_head():
    import torch
    from src.models.heads import ReconstructionHead
    from src.config import SpectralFMConfig
    
    cfg = SpectralFMConfig()
    head = ReconstructionHead(d_input=256, n_patches=cfg.n_patches, patch_size=cfg.patch_size)
    x = torch.randn(2, cfg.n_patches, 256)
    y = head(x)
    print(f"    ReconstructionHead output: {y.shape}")
    assert y.shape == (2, cfg.n_patches, cfg.patch_size)

def test_fno_head():
    import torch
    from src.models.heads import FNOTransferHead
    
    fno = FNOTransferHead(d_latent=128, out_channels=2048, width=64, modes=32, n_layers=4)
    z = torch.randn(2, 128)
    y = fno(z)
    print(f"    FNO output: {y.shape}")
    assert y.shape == (2, 2048), f"FNO output {y.shape} expected (2, 2048)"

def test_losses():
    import torch
    from src.losses.losses import MSRPLoss, PhysicsLoss
    
    # MSRP
    msrp = MSRPLoss()
    pred = torch.randn(2, 64, 32)
    target = torch.randn(2, 64, 32)
    mask = torch.zeros(2, 64)
    mask[:, :10] = 1  # Mask first 10 patches
    loss = msrp(pred, target, mask)
    assert loss.dim() == 0 and loss.item() > 0

def test_full_forward():
    import torch
    from src.config import SpectralFMConfig
    from src.models.spectral_fm import SpectralFM
    
    cfg = SpectralFMConfig()
    model = SpectralFM(cfg)
    x = torch.randn(2, 2048)
    
    # Encode
    enc = model.encode(x, domain="NIR")
    print(f"    Encode output keys: {list(enc.keys())}")
    print(f"    z_chem: {enc['z_chem'].shape}, z_inst: {enc['z_inst'].shape}")
    print(f"    tokens: {enc['tokens'].shape}")

def test_full_backward():
    import torch
    from src.config import SpectralFMConfig
    from src.models.spectral_fm import SpectralFM
    
    cfg = SpectralFMConfig()
    model = SpectralFM(cfg)
    x = torch.randn(2, 2048)
    
    enc = model.encode(x, domain="NIR")
    loss = enc["z_chem"].sum() + enc["moe_loss"]
    loss.backward()
    
    # Check gradients exist
    n_grad = sum(1 for p in model.parameters() if p.grad is not None)
    n_total = sum(1 for p in model.parameters())
    print(f"    Gradients: {n_grad}/{n_total} parameters have gradients")
    assert n_grad > 0, "No gradients!"

def test_pretrain_forward():
    import torch
    from src.config import SpectralFMConfig
    from src.models.spectral_fm import SpectralFM, SpectralFMForPretraining
    
    cfg = SpectralFMConfig()
    model = SpectralFM(cfg)
    pretrain_model = SpectralFMForPretraining(model, cfg)
    
    x = torch.randn(2, 2048)
    output = pretrain_model(x, domain="NIR")
    
    print(f"    Pretrain output keys: {list(output.keys())}")
    print(f"    Reconstruction: {output['reconstruction'].shape}")
    print(f"    Target patches: {output['target_patches'].shape}")
    print(f"    Mask: {output['mask'].shape}")

def test_ttt():
    import torch
    from src.config import SpectralFMConfig
    from src.models.spectral_fm import SpectralFM

    cfg = SpectralFMConfig()
    model = SpectralFM(cfg)

    test_spectra = torch.randn(10, 2048)
    model.test_time_train(test_spectra, n_steps=2, lr=1e-4)
    print("    TTT completed without error")


def test_wavelet_pywt():
    """Test that wavelet embedding uses pywt and produces correct shapes."""
    import torch
    from src.models.embedding import WaveletEmbedding

    emb = WaveletEmbedding(d_model=64, n_channels=2048, wavelet_levels=4,
                           patch_size=32, stride=16)
    x = torch.randn(2, 2048)
    tokens = emb(x)
    expected_n_patches = (2048 - 32) // 16 + 1  # 127
    expected_shape = (2, expected_n_patches + 2, 64)  # +2 for CLS + domain
    assert tokens.shape == expected_shape, f"Wrong shape: {tokens.shape}, expected {expected_shape}"
    assert torch.isfinite(tokens).all()
    print(f"    pywt wavelet output: {tokens.shape}")


def test_lora_injection():
    """Test LoRA injection and forward pass."""
    import torch
    from src.config import get_light_config
    from src.models.spectral_fm import SpectralFM
    from src.models.lora import inject_lora, get_lora_state_dict

    config = get_light_config()
    model = SpectralFM(config)
    total_before = sum(p.numel() for p in model.parameters())

    inject_lora(model, ["q_proj", "k_proj", "v_proj"], rank=4, alpha=8)

    total_after = sum(p.numel() for p in model.parameters())
    lora_params = sum(p.numel() for n, p in model.named_parameters() if 'lora_' in n)
    assert lora_params > 0, "No LoRA params found!"
    print(f"    LoRA params: {lora_params:,} (added {total_after - total_before:,})")

    # Forward still works
    x = torch.randn(2, 2048)
    model.eval()
    with torch.no_grad():
        out = model.encode(x)
    assert out["z_chem"].shape == (2, config.vib.z_chem_dim)

    # Freeze + LoRA stays trainable
    model.freeze_backbone()
    lora_trainable = sum(p.numel() for n, p in model.named_parameters()
                         if p.requires_grad and 'lora_' in n)
    assert lora_trainable > 0, "No LoRA params trainable after freeze"

    # State dict extraction
    lora_sd = get_lora_state_dict(model)
    assert len(lora_sd) > 0
    print(f"    LoRA state dict: {len(lora_sd)} keys")


def test_logger():
    """Test dual logging (JSON-only mode)."""
    import tempfile, json
    from src.utils.logging import ExperimentLogger

    with tempfile.TemporaryDirectory() as tmpdir:
        exp_logger = ExperimentLogger(
            project="test", run_name="smoke",
            use_wandb=False, log_dir=tmpdir
        )
        exp_logger.log({"loss": 1.0}, step=0)
        exp_logger.log({"loss": 0.5}, step=1)
        exp_logger.finish()

        log_file = f"{tmpdir}/smoke.jsonl"
        with open(log_file) as f:
            lines = f.readlines()
        assert len(lines) == 2, f"Expected 2 log lines, got {len(lines)}"
        entry = json.loads(lines[0])
        assert "loss" in entry
        assert "_step" in entry
    print("    JSON logging OK")


if __name__ == "__main__":
    print("=" * 60)
    print("SpectralFM v2 — Smoke Test Suite")
    print("=" * 60)
    
    print("\n--- 1. Imports ---")
    test("imports", test_imports)
    
    print("\n--- 2. Config ---")
    test("config", test_config)
    
    print("\n--- 3. Data ---")
    test("corn data", test_data_corn)
    test("tablet data", test_data_tablet)
    
    print("\n--- 4. Individual Modules ---")
    test("wavelet_embedding", test_wavelet_embedding)
    test("mamba", test_mamba)
    test("moe", test_moe)
    test("transformer", test_transformer)
    test("vib_head", test_vib_head)
    test("reconstruction_head", test_reconstruction_head)
    test("fno_head", test_fno_head)
    
    print("\n--- 5. Losses ---")
    test("losses", test_losses)
    
    print("\n--- 6. Full Model ---")
    test("full_forward", test_full_forward)
    test("full_backward", test_full_backward)
    test("pretrain_forward", test_pretrain_forward)
    
    print("\n--- 7. Test-Time Training ---")
    test("ttt", test_ttt)

    print("\n--- 8. P2: Architecture Fixes ---")
    test("wavelet_pywt", test_wavelet_pywt)
    test("lora_injection", test_lora_injection)
    test("logger", test_logger)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, s, _ in results if s == PASS)
    failed = sum(1 for _, s, _ in results if s == FAIL)
    print(f"  {PASS} Passed: {passed}")
    print(f"  {FAIL} Failed: {failed}")
    print(f"  Total: {passed + failed}")
    
    if failed > 0:
        print(f"\nFailed tests:")
        for name, status, err in results:
            if status == FAIL:
                print(f"  {FAIL} {name}: {err}")
    
    sys.exit(0 if failed == 0 else 1)
