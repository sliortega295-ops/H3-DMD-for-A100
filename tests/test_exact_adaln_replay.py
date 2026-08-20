from __future__ import annotations

import pytest
import torch

from h3_a100.adaln_cache import AdaLNCacheController
from h3_a100.exact_adaln_replay import install_exact_adaln_checkpoint_replay_scope


class _ToyTransformer:
    def __init__(self):
        self.gradient_checkpointing = True
        # Execute twice to model checkpoint original execution + replay.
        self._gradient_checkpointing_func = lambda fn, *args, **kwargs: (
            fn(*args, **kwargs),
            fn(*args, **kwargs),
        )[0]


def _controller_with_complete_key():
    controller = AdaLNCacheController(enabled=True, max_dynamic_keys=0)
    key = ("rollout", 3)
    controller._entries[key] = {
        0: (torch.tensor([1.0]),),
        1: (torch.tensor([2.0]),),
    }
    controller._persistent.add(key)
    return controller, key


def test_exact_replay_restores_key_and_never_computes_projection():
    transformer = _ToyTransformer()
    controller, key = _controller_with_complete_key()
    registration = install_exact_adaln_checkpoint_replay_scope(
        transformer, controller, expected_block_count=2
    )

    def projection_must_not_run(_):
        raise AssertionError("projection fallback executed")

    for block_index in (0, 1):
        def block(x, index=block_index):
            value = controller.get_or_compute(index, x, projection_must_not_run)[0]
            return x + value

        with controller.scope(key, persistent=True):
            transformer._gradient_checkpointing_func(block, torch.tensor([3.0]))

    assert registration.stats == {
        "checkpoint_wrap_count": 2,
        "scoped_execution_count": 4,
        "captured_key_missing_count": 0,
        "captured_key_incomplete_count": 0,
        "cache_hit_count": 4,
        "cache_miss_count": 0,
        "unexpected_hit_delta_count": 0,
    }


def test_exact_replay_fails_closed_without_active_key():
    transformer = _ToyTransformer()
    controller, _ = _controller_with_complete_key()
    install_exact_adaln_checkpoint_replay_scope(
        transformer, controller, expected_block_count=2
    )
    with pytest.raises(RuntimeError, match="no active cache key"):
        transformer._gradient_checkpointing_func(lambda x: x, torch.tensor([1.0]))
