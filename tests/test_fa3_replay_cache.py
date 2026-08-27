from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from h3_a100.fa3_nograd_splits import install_fa3_nograd_splits
from h3_a100.fa3_replay_cache import (
    EXPECTED_GRAD_TRANSFORMER_FORWARDS,
    FA3ReplayCacheRegistration,
    FA3ReplayCacheStats,
    install_fa3_replay_cache,
    parse_block_indices,
    validate_cycle,
)


ROOT = Path(__file__).resolve().parents[1]


def test_default_off_integration_and_parser():
    grid = (ROOT / "h3_a100" / "grid_adaln.py").read_text()
    loop = (ROOT / "h3_a100" / "trainer_loop.py").read_text()
    config = (
        ROOT / "configs" / "minimax_h3_t2av_dmd_a100_world16_grid1000.yaml"
    ).read_text()
    assert 'os.environ.get("H3_FA3_REPLAY_CACHE_BLOCKS")' in grid
    assert "install_fa3_replay_cache" in grid
    assert "_validate_fa3_replay_cache_cycle" in loop
    assert "fa3_replay_cache:" in config
    assert "enabled: false" in config
    assert parse_block_indices(None) == ()
    assert parse_block_indices("") == ()
    assert parse_block_indices("40-42,49") == (40, 41, 42, 49)
    with pytest.raises(ValueError, match="duplicate"):
        parse_block_indices("40,40")
    with pytest.raises(ValueError, match=r"\[0,49\]"):
        parse_block_indices("50")


def test_disabled_install_does_not_touch_transformer_or_kernel():
    checkpoint_fn = object()
    transformer = SimpleNamespace(_gradient_checkpointing_func=checkpoint_fn)
    kernel = SimpleNamespace(kernel_fn=lambda q, k, v, **kwargs: q)
    registration = install_fa3_replay_cache(
        transformer,
        block_indices=(),
        parent_split_registration=None,
        kernel_config=kernel,
    )
    assert not registration.enabled
    assert transformer._gradient_checkpointing_func is checkpoint_fn


class _Block(nn.Module):
    def __init__(self, kernel_config):
        super().__init__()
        self.kernel_config = kernel_config

    def forward(self, value):
        # sin() saves replay-relevant state, so every non-selected checkpoint
        # must also execute during backward instead of being optimized away by
        # non-reentrant early stop in this CPU contract test.
        return torch.sin(
            self.kernel_config.kernel_fn(value, value, value, num_splits=1)
        )


class _Transformer(nn.Module):
    def __init__(self, kernel_config):
        super().__init__()
        self.transformer_blocks = nn.ModuleList(
            [_Block(kernel_config) for _ in range(50)]
        )
        # Match pinned Diffusers: its closure deliberately accepts no kwargs,
        # so a candidate cannot accidentally rely on forwarding context_fn.
        def diffusers_checkpoint(module, *args):
            return checkpoint(module.__call__, *args, use_reentrant=False)

        self._gradient_checkpointing_func = diffusers_checkpoint

    def forward(self, value):
        for block in self.transformer_blocks:
            value = self._gradient_checkpointing_func(block, value)
        return value


def test_selected_blocks_cache_exact_compact_forward_and_keep_parent_census():
    def original(q, k, v, **kwargs):
        del k, v, kwargs
        return q.clone()

    def raw_forward(q, k, v, *args, **kwargs):
        del k, v, args, kwargs
        lse = torch.zeros(
            (q.shape[0], q.shape[2], q.shape[1]), device=q.device, dtype=torch.float32
        )
        return q.clone(), lse, torch.empty(0), torch.empty(0)

    def raw_backward(
        dout,
        q,
        k,
        v,
        out,
        lse,
        *args,
    ):
        del q, k, v, out, lse
        # The six optional slots precede dq/dk/dv in the pinned schema.
        dq, dk, dv = args[6:9]
        dq.copy_(dout)
        dk.zero_()
        dv.zero_()
        return torch.empty(0)

    kernel = SimpleNamespace(
        kernel_fn=original,
        wrapped_forward_fn=raw_forward,
        wrapped_backward_fn=raw_backward,
    )
    parent = install_fa3_nograd_splits(
        num_splits=2, grad_num_splits=2, kernel_config=kernel
    )
    transformer = _Transformer(kernel)
    registration = install_fa3_replay_cache(
        transformer,
        block_indices="48-49",
        parent_split_registration=parent,
        kernel_config=kernel,
        raw_forward=raw_forward,
        raw_backward=raw_backward,
    )
    # Match production ordering: the Grid replay/key wrapper is outside the
    # candidate and passes a scoped callable into it.
    candidate_checkpoint = transformer._gradient_checkpointing_func

    def grid_like_checkpoint(function, *args):
        def grid_scoped(*fn_args):
            return function(*fn_args)

        return candidate_checkpoint(grid_scoped, *args)

    transformer._gradient_checkpointing_func = grid_like_checkpoint

    initial = torch.randn(1, 2, 56, 128)
    reference = initial.clone().requires_grad_(True)
    reference_output = reference
    for _ in range(50):
        reference_output = torch.sin(reference_output)
    reference_output.sum().backward()

    value = initial.clone().requires_grad_(True)
    output = transformer(value)
    output.sum().backward()

    assert torch.equal(output, reference_output)
    assert torch.equal(value.grad, reference.grad)
    observed = registration.snapshot()
    assert observed["grad_transformer_forwards"] == 1
    assert observed["completed_grad_transformer_forwards"] == 1
    assert observed["selected_checkpoint_wraps"] == 2
    assert observed["selected_scoped_executions"] == 4
    assert observed["compact_attention_entries"] == 4
    assert observed["compact_forward_impl_calls"] == 2
    assert observed["compact_backward_calls"] == 2
    assert observed["unexpected_kernel_contract_calls"] == 0
    assert observed["unexpected_checkpoint_contract_calls"] == 0
    assert parent.stats.total_calls == 100
    assert parent.stats.grad_calls == 100
    assert parent.stats.grad_rewritten_calls == 100


def test_full_cycle_validator_is_fail_closed():
    blocks = (40, 41)
    selected = len(blocks) * EXPECTED_GRAD_TRANSFORMER_FORWARDS
    stats = FA3ReplayCacheStats(
        grad_transformer_forwards=6,
        completed_grad_transformer_forwards=6,
        selected_checkpoint_wraps=selected,
        selected_scoped_executions=2 * selected,
        compact_attention_entries=2 * selected,
        compact_forward_impl_calls=selected,
        compact_backward_calls=selected,
        cached_logical_bytes=123,
    )
    registration = FA3ReplayCacheRegistration(
        True, blocks, None, None, None, None, None, None, stats
    )
    assert validate_cycle(registration, {})["compact_backward_calls"] == selected
    stats.compact_backward_calls -= 1
    with pytest.raises(RuntimeError, match="compact_backward_calls"):
        validate_cycle(registration, {})
