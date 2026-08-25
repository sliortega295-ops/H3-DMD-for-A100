from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_fusion_is_opt_in_and_gradient_path_stays_reference():
    trainer = (ROOT / "h3_a100" / "trainer.py").read_text()
    fused = (ROOT / "h3_a100" / "fused_block_pointwise.py").read_text()
    config = (ROOT / "configs" / "minimax_h3_t2av_dmd_a100_world16_grid1000.yaml").read_text()

    assert '"H3_FUSED_BLOCK_POINTWISE"' in trainer
    assert "enabled: false" in config
    assert "if torch.is_grad_enabled():" in fused
    assert "return self._h3_original_forward" in fused
    assert "PINNED_BLOCK_FORWARD_SHA256" in fused
    assert "EXPECTED_NOGRAD_BLOCK_CALLS_PER_CYCLE" in fused


def test_a100_kernel_preserves_explicit_bf16_rounding_boundaries():
    source = (ROOT / "h3_a100" / "triton_fused_pointwise.py").read_text()
    assert "cvt.rn.bf16.f32" in source
    assert "cvt.f32.bf16" in source
    assert "pack=2" in source
    assert "requires SM80+" in source
    assert "torch.empty_like" in source

