from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from h3_a100.fused_qk_rmsnorm_rotary import (
    EXPECTED_NOGRAD_ATTENTION_CALLS_PER_CYCLE,
    EXPECTED_NOGRAD_QK_CALLS_PER_CYCLE,
    EXPECTED_REFERENCE_GRAD_ATTENTION_CALLS_PER_CYCLE,
    FusedQKRegistration,
    FusedQKStats,
    validate_cycle,
)
from h3_a100.fused_rotary import FusedRotaryRegistration, FusedRotaryStats


ROOT = Path(__file__).resolve().parents[1]


def test_qk_rmsnorm_rotary_is_independent_default_off_and_cycle_audited():
    trainer = (ROOT / "h3_a100" / "trainer.py").read_text()
    wrapper = (ROOT / "h3_a100" / "fused_qk_rmsnorm_rotary.py").read_text()
    kernel = (ROOT / "h3_a100" / "triton_fused_rotary.py").read_text()
    loop = (ROOT / "h3_a100" / "trainer_loop.py").read_text()
    config = (
        ROOT / "configs" / "minimax_h3_t2av_dmd_a100_world16_grid1000.yaml"
    ).read_text()

    assert '"H3_FUSED_QK_RMSNORM_ROTARY"' in trainer
    assert "requires H3_FUSED_ROTARY=1" in trainer
    assert "base_rotary_registration=self.fused_rotary_registration" in trainer
    assert "qk_rmsnorm: false" in config
    assert "if torch.is_grad_enabled():" in wrapper
    assert "if rotary_emb is None:" in wrapper
    assert "PINNED_PROCESSOR_CALL_SHA256" in wrapper
    assert "fused_qk_rmsnorm_rotary" in kernel
    assert "_validate_fused_qk_cycle" in loop
    assert "base_rotary_registration.stats.fused_nograd_calls += 2" in wrapper


def test_qk_feature_requires_live_base_rotary_registration():
    # The check happens before the pinned Diffusers import/model audit, making
    # this a cheap unit-level proof that the parent census cannot be bypassed.
    class Transformer:
        transformer_blocks = [object()] * 50

    from h3_a100.fused_qk_rmsnorm_rotary import install_fused_qk_rmsnorm_rotary

    with pytest.raises(RuntimeError, match="installed base rotary registration"):
        install_fused_qk_rmsnorm_rotary(Transformer(), enabled=True)

    disabled = FusedRotaryRegistration(False, False, None, FusedRotaryStats())
    with pytest.raises(RuntimeError, match="installed base rotary registration"):
        install_fused_qk_rmsnorm_rotary(
            Transformer(), enabled=True, base_rotary_registration=disabled
        )


def test_fused_call_forwards_physical_rotary_census(monkeypatch):
    import h3_a100.fused_qk_rmsnorm_rotary as feature
    import h3_a100.triton_fused_rotary as kernels

    class Processor:
        _attention_backend = None
        _parallel_config = None

        def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None):
            return hidden_states

    class Projection:
        def __call__(self, hidden):
            return torch.zeros((*hidden.shape[:-1], 56 * 128), dtype=hidden.dtype)

    class Attention:
        def __init__(self):
            self.processor = Processor()
            self.heads = 56
            self.head_dim = 128
            self.norm_q = torch.nn.RMSNorm(128, eps=1e-5)
            self.norm_k = torch.nn.RMSNorm(128, eps=1e-5)
            self.norm_q.weight.requires_grad_(False)
            self.norm_k.weight.requires_grad_(False)
            self.fused_projections = False
            self.to_q = Projection()
            self.to_k = Projection()
            self.to_v = Projection()
            self.to_out = [torch.nn.Identity(), torch.nn.Identity()]

    attention = Attention()
    transformer = SimpleNamespace(
        transformer_blocks=[SimpleNamespace(attn=attention) for _ in range(50)]
    )
    module = SimpleNamespace(
        MiniMaxH3AttnProcessor=Processor,
        dispatch_attention_fn=lambda query, key, value, **_kwargs: query,
    )
    monkeypatch.setattr(feature.importlib, "import_module", lambda _name: module)
    monkeypatch.setattr(
        feature,
        "PINNED_PROCESSOR_CALL_SHA256",
        feature._source_sha256(Processor.__call__),
    )
    monkeypatch.setattr(
        kernels,
        "fused_qk_rmsnorm_rotary",
        lambda hidden, _weight, _cos, _sin, *, eps: hidden,
    )
    base = FusedRotaryRegistration(True, True, "test", FusedRotaryStats())
    registration = feature.install_fused_qk_rmsnorm_rotary(
        transformer, enabled=True, base_rotary_registration=base
    )
    hidden = torch.zeros((1, 2, 32))
    rotary = (torch.zeros((2, 96)), torch.zeros((2, 96)))
    with torch.no_grad():
        output = attention.processor(attention, hidden, rotary, None)
    assert output.shape == (1, 2, 56 * 128)
    assert registration.snapshot() == {
        "fused_nograd_attention_calls": 1,
        "fused_nograd_qk_calls": 2,
        "reference_grad_attention_calls": 0,
    }
    assert base.stats.fused_nograd_calls == 2


def test_cycle_receipt_is_fail_closed():
    stats = FusedQKStats(
        fused_nograd_attention_calls=EXPECTED_NOGRAD_ATTENTION_CALLS_PER_CYCLE,
        fused_nograd_qk_calls=EXPECTED_NOGRAD_QK_CALLS_PER_CYCLE,
        reference_grad_attention_calls=EXPECTED_REFERENCE_GRAD_ATTENTION_CALLS_PER_CYCLE,
    )
    registration = FusedQKRegistration(
        enabled=True,
        source_sha256="test",
        attention_count=50,
        stats=stats,
    )
    delta = validate_cycle(registration, {})
    assert delta == {
        "fused_nograd_attention_calls": 1250,
        "fused_nograd_qk_calls": 2500,
        "reference_grad_attention_calls": 600,
    }

    stats.fused_nograd_qk_calls -= 1
    with pytest.raises(RuntimeError, match="expected=2500"):
        validate_cycle(registration, {})


def test_kernel_contract_is_bounded_to_production_geometry():
    source = (ROOT / "h3_a100" / "triton_fused_rotary.py").read_text()
    assert "B1/NH56/HD128" in source
    assert "rotary_dim=96" in source
    assert "eps=1e-5" in source
    assert "num_warps=2" in source
    assert "values * inverse_rms * weights" in source
