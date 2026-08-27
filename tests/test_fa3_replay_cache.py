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
    parse_max_d2h_inflight,
    parse_storage,
    prepare_staged_cache_before_backward,
    trim_allocator_before_backward,
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
    assert "storage: cuda" in config
    assert "max_d2h_inflight: 2" in config
    assert "trim_before_backward: false" in config
    assert "H3_FA3_REPLAY_CACHE_STORAGE" in grid
    assert "H3_FA3_REPLAY_CACHE_MAX_D2H_INFLIGHT" in grid
    assert "H3_FA3_REPLAY_CACHE_TRIM_BEFORE_BACKWARD" in grid
    assert parse_block_indices(None) == ()
    assert parse_block_indices("") == ()
    assert parse_block_indices("40-42,49") == (40, 41, 42, 49)
    with pytest.raises(ValueError, match="duplicate"):
        parse_block_indices("40,40")
    with pytest.raises(ValueError, match=r"\[0,49\]"):
        parse_block_indices("50")
    assert parse_storage(None) == "cuda"
    assert parse_storage("CPU") == "cpu"
    assert parse_storage("CPU_STAGED") == "cpu_staged"
    with pytest.raises(ValueError, match="storage must be"):
        parse_storage("disk")
    assert parse_max_d2h_inflight(None) == 2
    assert parse_max_d2h_inflight("1") == 1
    with pytest.raises(ValueError, match="must be >=1"):
        parse_max_d2h_inflight(0)


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


def test_allocator_trim_is_default_off_and_cpu_only(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("trim"))
    stats = SimpleNamespace(allocator_trim_calls=0)
    registration = SimpleNamespace(
        enabled=True,
        trim_before_backward=False,
        storage="cpu",
        stats=stats,
    )
    assert not trim_allocator_before_backward(registration)
    assert calls == []
    assert stats.allocator_trim_calls == 0

    registration.trim_before_backward = True
    assert trim_allocator_before_backward(registration)
    assert calls == ["trim"]
    assert stats.allocator_trim_calls == 1

    registration.storage = "cuda"
    with pytest.raises(RuntimeError, match="authorized only for CPU storage"):
        trim_allocator_before_backward(registration)

    registration.storage = "cpu_staged"
    assert trim_allocator_before_backward(registration)
    assert calls == ["trim", "trim"]


def test_staged_prefetch_gate_is_storage_scoped():
    calls = []
    pool = SimpleNamespace(prefetch_first_before_backward=lambda: calls.append("prefetch"))
    registration = SimpleNamespace(
        enabled=True,
        storage="cpu",
        staged_pool=pool,
        stats=SimpleNamespace(unexpected_cpu_storage_calls=0),
    )
    assert not prepare_staged_cache_before_backward(registration)
    assert calls == []
    registration.storage = "cpu_staged"
    assert prepare_staged_cache_before_backward(registration)
    assert calls == ["prefetch"]


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


def test_cpu_storage_cycle_validator_reconciles_exact_transfer_bytes():
    blocks = (0, 1)
    selected = len(blocks) * EXPECTED_GRAD_TRANSFORMER_FORWARDS
    logical_bytes = 456
    stats = FA3ReplayCacheStats(
        grad_transformer_forwards=6,
        completed_grad_transformer_forwards=6,
        selected_checkpoint_wraps=selected,
        selected_scoped_executions=2 * selected,
        compact_attention_entries=2 * selected,
        compact_forward_impl_calls=selected,
        compact_backward_calls=selected,
        cached_logical_bytes=logical_bytes,
        cpu_d2h_entries=selected,
        cpu_h2d_entries=selected,
        cpu_d2h_tensors=2 * selected,
        cpu_h2d_tensors=2 * selected,
        cpu_d2h_bytes=logical_bytes,
        cpu_h2d_bytes=logical_bytes,
    )
    registration = FA3ReplayCacheRegistration(
        True, blocks, None, None, None, None, None, None, stats, storage="cpu"
    )
    assert validate_cycle(registration, {})["cpu_h2d_bytes"] == logical_bytes
    stats.cpu_h2d_bytes -= 1
    with pytest.raises(RuntimeError, match="cpu_h2d_bytes"):
        validate_cycle(registration, {})


def test_staged_storage_cycle_validator_reconciles_host_staging():
    blocks = (38, 39)
    selected = len(blocks) * EXPECTED_GRAD_TRANSFORMER_FORWARDS
    logical_bytes = 789
    stats = FA3ReplayCacheStats(
        grad_transformer_forwards=6,
        completed_grad_transformer_forwards=6,
        selected_checkpoint_wraps=selected,
        selected_scoped_executions=2 * selected,
        compact_attention_entries=2 * selected,
        compact_forward_impl_calls=selected,
        compact_backward_calls=selected,
        cached_logical_bytes=logical_bytes,
        cpu_d2h_entries=selected,
        cpu_h2d_entries=selected,
        cpu_d2h_tensors=2 * selected,
        cpu_h2d_tensors=2 * selected,
        cpu_d2h_bytes=logical_bytes,
        cpu_h2d_bytes=logical_bytes,
        cpu_forward_host_copy_entries=selected,
        cpu_forward_host_copy_bytes=logical_bytes,
        cpu_backward_host_copy_entries=selected,
        cpu_backward_host_copy_bytes=logical_bytes,
        cpu_backward_prefetches=selected,
    )
    registration = FA3ReplayCacheRegistration(
        True, blocks, None, None, None, None, None, None, stats, storage="cpu_staged"
    )
    delta = validate_cycle(registration, {})
    assert delta["cpu_backward_prefetches"] == selected
    stats.cpu_backward_prefetch_misses += 1
    with pytest.raises(RuntimeError, match="cpu_backward_prefetch_misses"):
        validate_cycle(registration, {})
