#!/usr/bin/env python3
"""Minimal CPU unit test for the v4 in-repo LoRA implementation.

Covers: adapter forward equals base + (alpha/r) B A x; merged_state_dict emits the
plain (pre-injection) keyspace; folded weights reproduce eval-mode outputs; only
target projections are wrapped; base weights stay frozen. No model download.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
import torch.nn as nn

from train_pipeline_decisions_v4 import (LORA_TARGET_MODULES, inject_lora, is_lora_param,
                                         lora_linear_cls, merged_state_dict)


def tiny_model():
    """Fake DecisionModel-shaped module: .backbone with target + non-target Linears."""
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = nn.ModuleDict({n: nn.Linear(16, 16) for n in ("q_proj", "k_proj", "v_proj", "o_proj")})
            self.mlp = nn.ModuleDict({n: nn.Linear(16, 16) for n in ("gate_proj", "up_proj", "down_proj")})
            self.norm = nn.LayerNorm(16)  # non-Linear: must not be wrapped

    class FakeBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Block() for _ in range(2)])
            self.embed_tokens = nn.Embedding(32, 16)

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = FakeBackbone()
            self.scalar = nn.Linear(16, 1)

    torch.manual_seed(0)
    return FakeModel().eval()


def main():
    cls = lora_linear_cls()

    # 1) Adapter forward = base + scaling * B A x.
    base = nn.Linear(16, 16)
    wrapped = cls(base, rank=16, alpha=32.0, dropout=0.0).eval()
    x = torch.randn(5, 16)
    with torch.inference_mode():
        expect = base(x) + (x @ wrapped.lora_A.T @ wrapped.lora_B.T) * wrapped.scaling
        assert torch.allclose(wrapped(x), expect, atol=1e-7)

    # 2) Injection wraps exactly the 7 target projections per block; keys grow only by lora_*.
    model = tiny_model()
    plain_keys = set(model.state_dict())
    wrapped_names = inject_lora(model, rank=16, alpha=32.0, dropout=0.05)
    assert len(wrapped_names) == 2 * len(LORA_TARGET_MODULES) == 14
    assert all(any(name.endswith(t) for t in LORA_TARGET_MODULES) for name in wrapped_names)
    lora_keys = {k for k in model.state_dict() if is_lora_param(k)}
    assert len(lora_keys) == 2 * len(wrapped_names) == 28

    # 3) Base weights frozen; adapter params trainable.
    for name, p in model.named_parameters():
        if name.startswith("backbone."):
            assert p.requires_grad == is_lora_param(name), name

    # 4) Merged state dict: exact pre-injection keyspace; folded forward equals adapter
    #    forward in eval mode (dropout off) within fp32 tolerance.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if is_lora_param(name):
                nn.init.normal_(p, std=0.05)
    model.eval()  # injected modules default to train mode; equivalence holds with dropout off
    probes = [torch.randn(4, 16) for _ in range(3)]
    merged = merged_state_dict(model)
    assert set(merged) == plain_keys, "merged keyspace != pre-injection keyspace"
    with torch.inference_mode():
        for name in wrapped_names:
            mod = model.get_submodule("backbone." + name)
            merged_lin = nn.Linear(16, 16, bias=mod.linear.bias is not None)
            merged_lin.weight.data.copy_(merged["backbone." + name + ".weight"])
            if mod.linear.bias is not None:
                merged_lin.bias.data.copy_(merged["backbone." + name + ".bias"])
            for x in probes:
                delta = (mod(x) - merged_lin(x)).abs().max().item()
                assert delta <= 1e-6, f"{name}: merged forward diverges by {delta}"

    # 5) B = 0 at init → merged equals untouched base weights.
    model2 = tiny_model()
    before = {k: v.clone() for k, v in model2.state_dict().items()}
    inject_lora(model2, rank=16, alpha=32.0, dropout=0.0)
    merged2 = merged_state_dict(model2)
    for k, v in before.items():
        assert torch.equal(v, merged2[k]), f"zero-init merge changed {k}"

    print("test_lora_merge_v4: all checks passed "
          f"({len(wrapped_names)} wrapped modules, {len(lora_keys)} adapter params)")


if __name__ == "__main__":
    main()
