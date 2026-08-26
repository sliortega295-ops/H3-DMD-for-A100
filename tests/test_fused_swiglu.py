from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_fused_swiglu_is_opt_in_and_scoped_to_main_blocks():
    trainer = (ROOT / "h3_a100" / "trainer.py").read_text()
    fused = (ROOT / "h3_a100" / "fused_swiglu.py").read_text()
    config = (
        ROOT / "configs" / "minimax_h3_t2av_dmd_a100_world16_grid1000.yaml"
    ).read_text()

    assert '"H3_FUSED_SWIGLU"' in trainer
    assert '"H3_FUSED_SWIGLU_GRAD"' in trainer
    assert "swiglu_fusion:" in config
    assert "PINNED_SWIGLU_FORWARD_SHA256" in fused
    assert "transformer_blocks" in fused
    assert 'type(module).__name__ != "SwiGLU"' in fused
    assert "EXPECTED_NOGRAD_CALLS_PER_CYCLE" in fused
    assert "EXPECTED_GRAD_REPLAY_CALLS_PER_CYCLE" in fused
    assert "EXPECTED_BACKWARD_CALLS_PER_CYCLE" in fused


def test_fused_swiglu_kernel_keeps_projection_outside_candidate():
    kernel = (ROOT / "h3_a100" / "triton_fused_swiglu.py").read_text()
    fused = (ROOT / "h3_a100" / "fused_swiglu.py").read_text()

    assert "self.proj(hidden_states)" in fused
    assert "tl.sigmoid" in kernel
    assert "activated = (gates * sigmoid).to(tl.bfloat16)" in kernel
    assert "grad_activation = (gradients * values).to(tl.bfloat16)" in kernel
    assert "fused_swiglu_backward" in kernel
    assert "requires SM80+" in kernel
