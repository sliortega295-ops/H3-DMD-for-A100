from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_lora_epilogue_contract_is_fixed_and_nograd_only():
    kernel = (ROOT / "h3_a100" / "triton_lora_epilogue.py").read_text()
    forward = (ROOT / "h3_a100" / "lora_scale1_elision.py").read_text()
    trainer = (ROOT / "h3_a100" / "trainer.py").read_text()
    assert "LORA_RANK = 128" in kernel
    assert "BLOCK_M = 64" in kernel
    assert "BLOCK_N = 64" in kernel
    assert "base.dtype != torch.bfloat16" in kernel
    assert "projected.dtype != torch.bfloat16" in kernel
    assert "weight.dtype != torch.bfloat16" in kernel
    assert "rounded_projection = accumulator.to(tl.bfloat16).to(tl.float32)" in kernel
    assert "not grad_enabled" in forward
    assert "reference_grad_epilogue_calls" in forward
    assert "warmup_lora_epilogue" in trainer
