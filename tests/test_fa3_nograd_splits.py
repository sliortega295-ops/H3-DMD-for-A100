from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

from h3_a100.fa3_nograd_splits import (
    EXPECTED_GRAD_CALLS_PER_CYCLE,
    EXPECTED_NOGRAD_CALLS_PER_CYCLE,
    install_fa3_nograd_splits,
    validate_cycle,
)


ROOT = Path(__file__).resolve().parents[1]


def test_trainer_integration_is_default_off_and_cycle_audited():
    trainer = (ROOT / "h3_a100" / "trainer.py").read_text()
    loop = (ROOT / "h3_a100" / "trainer_loop.py").read_text()
    config = (
        ROOT / "configs" / "minimax_h3_t2av_dmd_a100_world16_grid1000.yaml"
    ).read_text()
    assert '"H3_FA3_NOGRAD_NUM_SPLITS"' in trainer
    assert "install_fa3_nograd_splits" in trainer
    assert "_validate_fa3_nograd_split_cycle" in loop
    assert "fa3_nograd_splits:" in config
    assert "num_splits: 1" in config


def test_default_is_disabled_and_does_not_touch_kernel():
    def original(**kwargs):
        return kwargs

    config = SimpleNamespace(kernel_fn=original)
    registration = install_fa3_nograd_splits(num_splits=1, kernel_config=config)
    assert not registration.enabled
    assert config.kernel_fn is original


def test_only_no_grad_calls_are_rewritten():
    observed = []

    def original(**kwargs):
        observed.append(kwargs["num_splits"])
        return kwargs["num_splits"]

    config = SimpleNamespace(kernel_fn=original)
    registration = install_fa3_nograd_splits(num_splits=2, kernel_config=config)
    with torch.no_grad():
        assert config.kernel_fn(num_splits=1) == 2
    with torch.enable_grad():
        assert config.kernel_fn(num_splits=1) == 1
    assert observed == [2, 1]
    assert registration.snapshot() == {
        "total_calls": 2,
        "no_grad_calls": 1,
        "grad_calls": 1,
        "rewritten_calls": 1,
        "unexpected_input_num_splits": 0,
    }


def test_cycle_contract_and_fail_closed_input():
    config = SimpleNamespace(kernel_fn=lambda **kwargs: kwargs["num_splits"])
    registration = install_fa3_nograd_splits(num_splits=2, kernel_config=config)
    registration.stats.total_calls = EXPECTED_NOGRAD_CALLS_PER_CYCLE + EXPECTED_GRAD_CALLS_PER_CYCLE
    registration.stats.no_grad_calls = EXPECTED_NOGRAD_CALLS_PER_CYCLE
    registration.stats.grad_calls = EXPECTED_GRAD_CALLS_PER_CYCLE
    registration.stats.rewritten_calls = EXPECTED_NOGRAD_CALLS_PER_CYCLE
    delta = validate_cycle(registration, {})
    assert delta["rewritten_calls"] == EXPECTED_NOGRAD_CALLS_PER_CYCLE
    with torch.no_grad(), pytest.raises(RuntimeError, match="expected Diffusers"):
        config.kernel_fn(num_splits=4)


def test_rejects_unregistered_candidates_and_double_install():
    with pytest.raises(ValueError, match="only authorizes"):
        install_fa3_nograd_splits(
            num_splits=4, kernel_config=SimpleNamespace(kernel_fn=lambda **kwargs: None)
        )
    config = SimpleNamespace(kernel_fn=lambda **kwargs: None)
    install_fa3_nograd_splits(num_splits=2, kernel_config=config)
    with pytest.raises(RuntimeError, match="more than once"):
        install_fa3_nograd_splits(num_splits=2, kernel_config=config)
