from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from h3_a100.lora_scale1_elision import (
    EXPECTED_DISABLED_CALLS_PER_CYCLE,
    EXPECTED_GRAD_ELISIONS_PER_CYCLE,
    EXPECTED_MODULE_COUNT,
    EXPECTED_NOGRAD_ELISIONS_PER_CYCLE,
    EXPECTED_TOTAL_CALLS_PER_CYCLE,
    install_lora_scale1_elision,
)


ROOT = Path(__file__).resolve().parents[1]


def test_lora_scale1_elision_is_default_off_and_fail_closed():
    trainer = (ROOT / "h3_a100" / "trainer.py").read_text()
    module = (ROOT / "h3_a100" / "lora_scale1_elision.py").read_text()
    config = (
        ROOT / "configs" / "minimax_h3_t2av_dmd_a100_world16_grid1000.yaml"
    ).read_text()
    assert '"H3_LORA_SCALE1_ELISION"' in trainer
    assert '"H3_LORA_NOGRAD_EPILOGUE"' in trainer
    assert '"H3_LORA_GRAD_EPILOGUE"' in trainer
    assert "lora_scale1_elision:" in config
    assert "enabled: false" in config
    assert "fused_nograd_epilogue: false" in config
    assert "fused_grad_epilogue: false" in config
    assert "PINNED_PEFT_LINEAR_FORWARD_SHA256" in module
    assert "EXPECTED_MODULE_COUNT = 52 * 6" in module
    assert EXPECTED_MODULE_COUNT == 312
    assert EXPECTED_TOTAL_CALLS_PER_CYCLE == 11544
    assert EXPECTED_DISABLED_CALLS_PER_CYCLE == 312
    assert EXPECTED_NOGRAD_ELISIONS_PER_CYCLE == 7488
    assert EXPECTED_GRAD_ELISIONS_PER_CYCLE == 3744


def _make_model():
    peft = pytest.importorskip("peft")
    from peft import LoraConfig

    base = torch.nn.Sequential(torch.nn.Linear(32, 48, bias=False))
    config = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["0"],
        bias="none",
    )
    return peft.get_peft_model(base, config)


def test_lora_scale1_elision_forward_and_gradients_are_bitwise_equal(monkeypatch):
    from h3_a100 import lora_scale1_elision as candidate

    # The toy has one PEFT Linear rather than the production 312; keep the
    # production installer/counter logic while bounding this parity test.
    monkeypatch.setattr(candidate, "EXPECTED_MODULE_COUNT", 1)
    reference = _make_model().to(dtype=torch.bfloat16)
    optimized = copy.deepcopy(reference)
    registration = install_lora_scale1_elision(optimized, enabled=True)
    assert registration.module_count == 1

    x_ref = torch.randn(4, 7, 32, dtype=torch.bfloat16, requires_grad=True)
    x_opt = x_ref.detach().clone().requires_grad_(True)
    out_ref = reference(x_ref)
    out_opt = optimized(x_opt)
    assert torch.equal(out_ref, out_opt)
    grad = torch.randn_like(out_ref)
    out_ref.backward(grad)
    out_opt.backward(grad)
    assert torch.equal(x_ref.grad, x_opt.grad)
    ref_params = dict(reference.named_parameters())
    opt_params = dict(optimized.named_parameters())
    for name in ref_params:
        if ref_params[name].grad is not None:
            assert torch.equal(ref_params[name].grad, opt_params[name].grad), name
    assert registration.snapshot() == {
        "total_calls": 1,
        "elided_calls": 1,
        "no_grad_elided_calls": 0,
        "grad_elided_calls": 1,
        "disabled_reference_calls": 0,
        "unsupported_reference_calls": 0,
        "invalid_contract_calls": 0,
        "fused_nograd_epilogue_calls": 0,
        "fused_grad_epilogue_calls": 0,
        "fused_grad_epilogue_backward_calls": 0,
        "reference_grad_epilogue_calls": 0,
    }


def test_lora_nograd_epilogue_is_selected_only_without_grad(monkeypatch):
    from h3_a100 import lora_scale1_elision as candidate

    monkeypatch.setattr(candidate, "EXPECTED_MODULE_COUNT", 1)
    reference = _make_model().to(dtype=torch.bfloat16)
    optimized = copy.deepcopy(reference)
    registration = install_lora_scale1_elision(
        optimized, enabled=True, epilogue_enabled=True
    )
    calls = []

    def fake_epilogue(base, projected, weight):
        calls.append((base.dtype, projected.dtype, weight.dtype))
        return (base + torch.nn.functional.linear(projected, weight)).to(base.dtype)

    monkeypatch.setattr(candidate, "fused_lora_b_residual", fake_epilogue)
    value = torch.randn(4, 7, 32, dtype=torch.bfloat16)
    with torch.no_grad():
        expected = reference(value)
        observed = optimized(value)
    assert torch.equal(expected, observed)
    assert calls == [(torch.bfloat16, torch.bfloat16, torch.bfloat16)]

    calls.clear()
    grad_value = value.detach().clone().requires_grad_(True)
    optimized(grad_value).sum().backward()
    assert calls == []
    assert registration.stats.fused_nograd_epilogue_calls == 1
    assert registration.stats.reference_grad_epilogue_calls == 1


def test_lora_nograd_epilogue_requires_scale_one(monkeypatch):
    from h3_a100 import lora_scale1_elision as candidate

    monkeypatch.setattr(candidate, "EXPECTED_MODULE_COUNT", 1)
    model = _make_model().to(dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="requires scale-one elision"):
        install_lora_scale1_elision(model, enabled=False, epilogue_enabled=True)


def test_lora_grad_epilogue_matches_reference_forward_and_backward(monkeypatch):
    from h3_a100 import lora_scale1_elision as candidate

    monkeypatch.setattr(candidate, "EXPECTED_MODULE_COUNT", 1)
    reference = _make_model().to(dtype=torch.bfloat16)
    optimized = copy.deepcopy(reference)
    registration = install_lora_scale1_elision(
        optimized,
        enabled=True,
        epilogue_enabled=True,
        grad_epilogue_enabled=True,
    )

    def fake_epilogue(base, projected, weight):
        return (base + torch.nn.functional.linear(projected, weight)).to(base.dtype)

    monkeypatch.setattr(candidate, "fused_lora_b_residual", fake_epilogue)
    x_ref = torch.randn(4, 7, 32, dtype=torch.bfloat16, requires_grad=True)
    x_opt = x_ref.detach().clone().requires_grad_(True)
    out_ref = reference(x_ref)
    out_opt = optimized(x_opt)
    assert torch.equal(out_ref, out_opt)
    grad = torch.randn_like(out_ref)
    out_ref.backward(grad)
    out_opt.backward(grad)
    assert torch.equal(x_ref.grad, x_opt.grad)
    ref_params = dict(reference.named_parameters())
    opt_params = dict(optimized.named_parameters())
    for name in ref_params:
        if ref_params[name].grad is not None:
            assert torch.equal(ref_params[name].grad, opt_params[name].grad), name
    assert registration.stats.fused_grad_epilogue_calls == 1
    assert registration.stats.fused_grad_epilogue_backward_calls == 1
    assert registration.stats.reference_grad_epilogue_calls == 0


def test_lora_grad_epilogue_requires_base_epilogue(monkeypatch):
    from h3_a100 import lora_scale1_elision as candidate

    monkeypatch.setattr(candidate, "EXPECTED_MODULE_COUNT", 1)
    model = _make_model().to(dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="requires base LoRA epilogue"):
        install_lora_scale1_elision(
            model, enabled=True, epilogue_enabled=False, grad_epilogue_enabled=True
        )


def test_lora_scale1_elision_rejects_non_identity_scale(monkeypatch):
    from h3_a100 import lora_scale1_elision as candidate

    monkeypatch.setattr(candidate, "EXPECTED_MODULE_COUNT", 1)
    model = _make_model().to(dtype=torch.bfloat16)
    registration = install_lora_scale1_elision(model, enabled=True)
    linear = next(module for module in model.modules() if type(module).__name__ == "Linear")
    linear.scaling[linear.active_adapters[0]] = 0.5
    with pytest.raises(RuntimeError, match="non-identity scaling"):
        model(torch.randn(2, 32, dtype=torch.bfloat16))
    assert registration.stats.invalid_contract_calls == 1


def test_lora_scale1_elision_counts_entry_before_checkpoint_early_stop(monkeypatch):
    from h3_a100 import lora_scale1_elision as candidate

    monkeypatch.setattr(candidate, "EXPECTED_MODULE_COUNT", 1)
    model = _make_model().to(dtype=torch.bfloat16)
    registration = install_lora_scale1_elision(model, enabled=True)
    linear = next(module for module in model.modules() if type(module).__name__ == "Linear")

    class EarlyStop(BaseException):
        pass

    def stop(_value):
        raise EarlyStop

    linear.base_layer.forward = stop
    with pytest.raises(EarlyStop):
        model(torch.randn(2, 32, dtype=torch.bfloat16))
    assert registration.stats.total_calls == 1
    assert registration.stats.elided_calls == 1
    assert registration.stats.grad_elided_calls == 1
