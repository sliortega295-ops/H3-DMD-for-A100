from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_lora_epilogue_contract_is_fixed_and_grad_path_is_opt_in():
    kernel = (ROOT / "h3_a100" / "triton_lora_epilogue.py").read_text()
    forward = (ROOT / "h3_a100" / "lora_scale1_elision.py").read_text()
    trainer = (ROOT / "h3_a100" / "trainer.py").read_text()
    assert "LORA_RANK = 128" in kernel
    assert "BLOCK_M = 64" in kernel
    assert "DEFAULT_BLOCK_N = 64" in kernel
    assert 'H3_LORA_EPILOGUE_BLOCK_N' in kernel
    assert "BLOCK_N not in {64, 128}" in kernel
    assert "base.dtype != torch.bfloat16" in kernel
    assert "projected.dtype != torch.bfloat16" in kernel
    assert "weight.dtype != torch.bfloat16" in kernel
    assert "rounded_projection = accumulator.to(tl.bfloat16).to(tl.float32)" in kernel
    assert "_FusedLoRABResidualAutograd" in forward
    assert '"H3_LORA_GRAD_EPILOGUE"' in trainer
    assert "fused_grad_epilogue_calls" in forward
    assert "fused_grad_epilogue_backward_calls" in forward
    assert "reference_grad_epilogue_calls" in forward
    assert "warmup_lora_epilogue" in trainer
    assert '"epilogue_block_n": LORA_EPILOGUE_BLOCK_N' in forward
    assert '"epilogue={} epilogue_block_n={} delta={}"' in (
        ROOT / "h3_a100" / "trainer_loop.py"
    ).read_text()
