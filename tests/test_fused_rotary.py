from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_rotary_fusion_is_opt_in_and_grad_path_stays_reference():
    trainer = (ROOT / "h3_a100" / "trainer.py").read_text()
    fused = (ROOT / "h3_a100" / "fused_rotary.py").read_text()
    config = (ROOT / "configs" / "minimax_h3_t2av_dmd_a100_world16_grid1000.yaml").read_text()

    assert '"H3_FUSED_ROTARY"' in trainer
    assert "rotary_fusion:" in config
    assert "if torch.is_grad_enabled():" in fused
    assert "return original(hidden_states, cos, sin)" in fused
    assert "PINNED_ROTARY_SOURCE_SHA256" in fused
    assert "EXPECTED_FUSED_NOGRAD_CALLS_PER_CYCLE" in fused
    assert "EXPECTED_REFERENCE_GRAD_CALLS_PER_CYCLE" in fused


def test_rotary_kernel_preserves_explicit_bf16_rounding_boundaries():
    source = (ROOT / "h3_a100" / "triton_fused_rotary.py").read_text()
    assert "cvt.rn.bf16.f32" in source
    assert "cvt.f32.bf16" in source
    assert "pack=2" in source
    assert "requires SM80+" in source
    assert "torch.empty_like" in source
    assert "rotary_columns < half" in source
    assert "n_elements: tl.constexpr" not in source
    assert "sequence_length: tl.constexpr" not in source
    assert "frozen B1 contract" in source
