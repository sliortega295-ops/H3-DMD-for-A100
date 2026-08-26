from pathlib import Path

import pytest
import torch

from h3_a100.fused_block_pointwise import (
    EXPECTED_NOGRAD_RMSNORM_MODULATION_CALLS_PER_CYCLE,
    FusedPointwiseRegistration,
    FusedPointwiseStats,
    install_fused_block_pointwise,
    validate_cycle,
)


ROOT = Path(__file__).resolve().parents[1]


def test_rmsnorm_modulation_is_independent_default_off_and_nograd_only():
    trainer = (ROOT / "h3_a100" / "trainer.py").read_text()
    forward = (ROOT / "h3_a100" / "fused_block_pointwise.py").read_text()
    kernel = (ROOT / "h3_a100" / "triton_fused_pointwise.py").read_text()
    config = (
        ROOT / "configs" / "minimax_h3_t2av_dmd_a100_world16_grid1000.yaml"
    ).read_text()

    assert '"H3_FUSED_RMSNORM_MODULATE"' in trainer
    assert "rmsnorm_modulate: false" in config
    assert "not grad_enabled and self._h3_fused_rmsnorm_modulate_enabled" in forward
    assert "EXPECTED_NOGRAD_RMSNORM_MODULATION_CALLS_PER_CYCLE" in forward
    assert "values * inverse_rms * weights" in kernel
    assert ".to(tl.bfloat16)" in kernel
    assert "_bf16_modulate_exact(normalized, scales, shifts)" in kernel
    assert "requires hidden_size=5376" in kernel
    assert "preregistered for B1" in kernel


def test_rmsnorm_modulation_requires_base_pointwise_fusion():
    with pytest.raises(RuntimeError, match="requires base pointwise fusion"):
        install_fused_block_pointwise(
            torch.nn.Module(),
            enabled=False,
            grad_enabled=False,
            rmsnorm_modulate_enabled=True,
        )


def test_cycle_receipt_counts_exactly_2500_nograd_calls():
    stats = FusedPointwiseStats(
        fused_nograd_block_calls=1250,
        fused_grad_block_calls=600,
        fused_modulation_calls=2500,
        fused_residual_calls=2500,
        fused_grad_modulation_calls=1200,
        fused_grad_residual_calls=1200,
        fused_grad_modulation_backward_calls=600,
        fused_grad_residual_backward_calls=600,
        fused_nograd_rmsnorm_modulation_calls=(
            EXPECTED_NOGRAD_RMSNORM_MODULATION_CALLS_PER_CYCLE
        ),
    )
    registration = FusedPointwiseRegistration(
        enabled=True,
        grad_enabled=True,
        rmsnorm_modulate_enabled=True,
        source_sha256="test",
        block_count=50,
        stats=stats,
    )
    delta = validate_cycle(registration, {})
    assert delta["fused_nograd_rmsnorm_modulation_calls"] == 2500

    stats.fused_nograd_rmsnorm_modulation_calls -= 1
    with pytest.raises(RuntimeError, match="expected=2500"):
        validate_cycle(registration, {})

