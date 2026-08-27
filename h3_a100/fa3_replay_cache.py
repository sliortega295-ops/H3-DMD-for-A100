"""Exact compact FA3 replay cache for selected native checkpoint blocks.

The pinned split2 custom op returns large accumulation workspaces in addition
to the public output/LSE pair.  PyTorch selective checkpointing caches an
operation's entire return tree, so marking that raw op would retain about
2.54 GiB per block.  This module exposes an opaque custom op whose public
contract contains only the exact output and softmax LSE.  Its implementation
still calls the same pinned split2 forward, and its autograd function calls the
same pinned backward.

The feature is default-off and only active inside explicitly selected native
per-block checkpoint calls.  It does not change no-grad attention, checkpoint
boundaries, FSDP placement, or application-level call counts.
"""

from __future__ import annotations

import contextvars
import dataclasses
import functools
import math
from collections import defaultdict, deque
from collections.abc import Callable, Iterable
from typing import Any

import torch
from loguru import logger
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.checkpoint import checkpoint, create_selective_checkpoint_contexts


EXPECTED_BLOCK_COUNT = 50
EXPECTED_GRAD_TRANSFORMER_FORWARDS = 6
EXPECTED_HEADS = 56
EXPECTED_HEAD_DIM = 128
COMPACT_FORWARD_SCHEMA = (
    "h3_a100::fa3_forward_compact(Tensor q, Tensor k, Tensor v, "
    "float softmax_scale) -> (Tensor, Tensor)"
)
SUPPORTED_STORAGE = ("cuda", "cpu")


@dataclasses.dataclass
class FA3ReplayCacheStats:
    grad_transformer_forwards: int = 0
    completed_grad_transformer_forwards: int = 0
    selected_checkpoint_wraps: int = 0
    selected_scoped_executions: int = 0
    compact_attention_entries: int = 0
    compact_forward_impl_calls: int = 0
    compact_backward_calls: int = 0
    cached_logical_bytes: int = 0
    cpu_d2h_entries: int = 0
    cpu_h2d_entries: int = 0
    cpu_d2h_tensors: int = 0
    cpu_h2d_tensors: int = 0
    cpu_d2h_bytes: int = 0
    cpu_h2d_bytes: int = 0
    cpu_pool_allocated_bytes: int = 0
    cpu_pool_reused_tensors: int = 0
    cpu_pool_busy_misses: int = 0
    unexpected_cpu_storage_calls: int = 0
    unexpected_kernel_contract_calls: int = 0
    unexpected_checkpoint_contract_calls: int = 0


@dataclasses.dataclass
class _Runtime:
    raw_forward: Callable[..., Any]
    raw_backward: Callable[..., Any]
    stats: FA3ReplayCacheStats


_RUNTIME: _Runtime | None = None
_CACHE_ACTIVE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "h3_a100_fa3_replay_cache_active", default=False
)


@torch.library.custom_op("h3_a100::fa3_forward_compact", mutates_args=())
def _fa3_forward_compact(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    runtime = _RUNTIME
    if runtime is None:
        raise RuntimeError("H3 compact FA3 runtime was used before installation")
    out, softmax_lse, *_workspace = runtime.raw_forward(
        q,
        k,
        v,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        float(softmax_scale),
        causal=False,
        window_size_left=-1,
        window_size_right=-1,
        attention_chunk=0,
        softcap=0.0,
        num_splits=2,
        pack_gqa=None,
        sm_margin=0,
    )
    runtime.stats.compact_forward_impl_calls += 1
    runtime.stats.cached_logical_bytes += int(
        out.numel() * out.element_size()
        + softmax_lse.numel() * softmax_lse.element_size()
    )
    return out, softmax_lse


@_fa3_forward_compact.register_fake
def _(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del k, v, softmax_scale
    return torch.empty_like(q), torch.empty(
        (q.shape[0], q.shape[2], q.shape[1]),
        dtype=torch.float32,
        device=q.device,
    )


class _CompactFA3Autograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float):
        out, softmax_lse = _fa3_forward_compact(q, k, v, float(scale))
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.scale = float(scale)
        return out

    @staticmethod
    def backward(ctx: Any, dout: torch.Tensor):
        runtime = _RUNTIME
        if runtime is None:
            raise RuntimeError("H3 compact FA3 backward ran before installation")
        q, k, v, out, softmax_lse = ctx.saved_tensors
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        runtime.raw_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            None,
            None,
            None,
            None,
            None,
            None,
            dq,
            dk,
            dv,
            ctx.scale,
            False,
            -1,
            -1,
            0.0,
            False,
            0,
        )
        runtime.stats.compact_backward_calls += 1
        return dq, dk, dv, None


@dataclasses.dataclass
class _PinnedBuffer:
    tensor: torch.Tensor
    reusable_after: torch.cuda.Event | None = None


@dataclasses.dataclass
class _CPUCacheEntry:
    buffers: tuple[_PinnedBuffer, ...]
    ready: torch.cuda.Event
    device: torch.device


class _PinnedPool:
    """Process-local pinned buffers reused across the six sequential grad graphs."""

    def __init__(self, stats: FA3ReplayCacheStats) -> None:
        self.stats = stats
        self._buffers: dict[tuple[Any, ...], list[_PinnedBuffer]] = defaultdict(list)
        self._copy_streams: dict[int, torch.cuda.Stream] = {}

    @staticmethod
    def _key(block_index: int, tensor_index: int, value: torch.Tensor) -> tuple[Any, ...]:
        return (
            int(block_index),
            int(tensor_index),
            tuple(value.shape),
            tuple(value.stride()),
            value.dtype,
        )

    def copy_stream(self, device: torch.device) -> torch.cuda.Stream:
        index = int(device.index if device.index is not None else torch.cuda.current_device())
        stream = self._copy_streams.get(index)
        if stream is None:
            stream = torch.cuda.Stream(device=index)
            self._copy_streams[index] = stream
        return stream

    def acquire(
        self, block_index: int, tensor_index: int, value: torch.Tensor
    ) -> _PinnedBuffer:
        key = self._key(block_index, tensor_index, value)
        candidates = self._buffers[key]
        for candidate in candidates:
            event = candidate.reusable_after
            if event is None or event.query():
                candidate.reusable_after = None
                self.stats.cpu_pool_reused_tensors += 1
                return candidate
        if candidates:
            self.stats.cpu_pool_busy_misses += 1
        tensor = torch.empty_like(
            value, device="cpu", pin_memory=True, memory_format=torch.preserve_format
        )
        candidate = _PinnedBuffer(tensor=tensor)
        candidates.append(candidate)
        self.stats.cpu_pool_allocated_bytes += int(
            tensor.numel() * tensor.element_size()
        )
        return candidate

    @staticmethod
    def release(buffer: _PinnedBuffer, reusable_after: torch.cuda.Event) -> None:
        buffer.reusable_after = reusable_after


class _HostCachingMode(TorchDispatchMode):
    def __init__(
        self,
        storage: deque[_CPUCacheEntry],
        pool: _PinnedPool,
        stats: FA3ReplayCacheStats,
        block_index: int,
    ) -> None:
        super().__init__()
        self.storage = storage
        self.pool = pool
        self.stats = stats
        self.block_index = int(block_index)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        del types
        kwargs = {} if kwargs is None else kwargs
        result = func(*args, **kwargs)
        if func != _fa3_forward_compact._opoverload:
            return result
        values = tuple(result)
        if len(values) != 2 or any(value.device.type != "cuda" for value in values):
            self.stats.unexpected_cpu_storage_calls += 1
            raise RuntimeError("H3 CPU FA3 cache expected two CUDA output tensors")
        device = values[0].device
        if any(value.device != device for value in values):
            self.stats.unexpected_cpu_storage_calls += 1
            raise RuntimeError("H3 CPU FA3 cache outputs span multiple CUDA devices")

        buffers = tuple(
            self.pool.acquire(self.block_index, index, value)
            for index, value in enumerate(values)
        )
        producer = torch.cuda.Event()
        producer.record(torch.cuda.current_stream(device))
        copy_stream = self.pool.copy_stream(device)
        ready = torch.cuda.Event()
        with torch.cuda.stream(copy_stream):
            copy_stream.wait_event(producer)
            for buffer, value in zip(buffers, values):
                buffer.tensor.copy_(value, non_blocking=True)
                value.record_stream(copy_stream)
            ready.record(copy_stream)
        logical_bytes = sum(
            int(buffer.tensor.numel() * buffer.tensor.element_size())
            for buffer in buffers
        )
        self.storage.append(_CPUCacheEntry(buffers, ready, device))
        self.stats.cpu_d2h_entries += 1
        self.stats.cpu_d2h_tensors += len(buffers)
        self.stats.cpu_d2h_bytes += logical_bytes
        return result


class _HostCachedMode(TorchDispatchMode):
    def __init__(
        self,
        storage: deque[_CPUCacheEntry],
        pool: _PinnedPool,
        stats: FA3ReplayCacheStats,
    ) -> None:
        super().__init__()
        self.storage = storage
        self.pool = pool
        self.stats = stats

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        del types
        kwargs = {} if kwargs is None else kwargs
        if func != _fa3_forward_compact._opoverload:
            return func(*args, **kwargs)
        if not self.storage:
            self.stats.unexpected_cpu_storage_calls += 1
            raise RuntimeError("H3 CPU FA3 replay had no staged host entry")
        entry = self.storage.popleft()
        stream = torch.cuda.current_stream(entry.device)
        stream.wait_event(entry.ready)
        values = tuple(
            torch.empty_like(buffer.tensor, device=entry.device)
            for buffer in entry.buffers
        )
        for value, buffer in zip(values, entry.buffers):
            value.copy_(buffer.tensor, non_blocking=True)
        reusable_after = torch.cuda.Event()
        reusable_after.record(stream)
        logical_bytes = sum(
            int(buffer.tensor.numel() * buffer.tensor.element_size())
            for buffer in entry.buffers
        )
        for buffer in entry.buffers:
            self.pool.release(buffer, reusable_after)
        self.stats.cpu_h2d_entries += 1
        self.stats.cpu_h2d_tensors += len(entry.buffers)
        self.stats.cpu_h2d_bytes += logical_bytes
        return values


def _create_cpu_cache_contexts(
    registration: "FA3ReplayCacheRegistration", block_index: int
) -> tuple[TorchDispatchMode, TorchDispatchMode]:
    if registration.cpu_pool is None:
        registration.stats.unexpected_cpu_storage_calls += 1
        raise RuntimeError("H3 CPU FA3 cache has no pinned-buffer pool")
    storage: deque[_CPUCacheEntry] = deque()
    return (
        _HostCachingMode(storage, registration.cpu_pool, registration.stats, block_index),
        _HostCachedMode(storage, registration.cpu_pool, registration.stats),
    )


@dataclasses.dataclass
class FA3ReplayCacheRegistration:
    enabled: bool
    block_indices: tuple[int, ...]
    transformer: torch.nn.Module | None
    original_checkpoint: Callable[..., Any] | None
    original_kernel: Callable[..., Any] | None
    parent_split_registration: Any | None
    pre_hook: Any | None
    post_hook: Any | None
    stats: FA3ReplayCacheStats
    raw_forward_schema: str | None = None
    raw_backward_schema: str | None = None
    storage: str = "cuda"
    cpu_pool: _PinnedPool | None = None

    def snapshot(self) -> dict[str, int]:
        return dataclasses.asdict(self.stats)

    def receipt(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "block_indices": list(self.block_indices),
            "compact_forward_schema": COMPACT_FORWARD_SCHEMA,
            "raw_forward_schema": self.raw_forward_schema,
            "raw_backward_schema": self.raw_backward_schema,
            "storage": self.storage,
            "stats": self.snapshot(),
        }


def parse_block_indices(value: str | Iterable[int] | None) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return ()
        parsed: list[int] = []
        for field in value.split(","):
            field = field.strip()
            if not field:
                continue
            if "-" in field:
                first, last = (int(part) for part in field.split("-", 1))
                if last < first:
                    raise ValueError(f"descending FA3 replay-cache range: {field}")
                parsed.extend(range(first, last + 1))
            else:
                parsed.append(int(field))
    else:
        parsed = [int(item) for item in value]
    result = tuple(sorted(set(parsed)))
    if len(result) != len(parsed):
        raise ValueError(f"duplicate FA3 replay-cache blocks: {parsed}")
    if any(index < 0 or index >= EXPECTED_BLOCK_COUNT for index in result):
        raise ValueError(f"FA3 replay-cache blocks must be in [0,49], got {result}")
    return result


def parse_storage(value: str | None) -> str:
    storage = "cuda" if value is None else str(value).strip().lower()
    if storage not in SUPPORTED_STORAGE:
        raise ValueError(
            f"FA3 replay-cache storage must be one of {SUPPORTED_STORAGE}, got {value!r}"
        )
    return storage


def _schema(value: Any) -> str | None:
    schema = getattr(value, "_schema", None)
    return None if schema is None else str(schema)


def install_fa3_replay_cache(
    transformer: torch.nn.Module,
    *,
    block_indices: str | Iterable[int] | None,
    parent_split_registration: Any,
    kernel_config: Any | None = None,
    raw_forward: Callable[..., Any] | None = None,
    raw_backward: Callable[..., Any] | None = None,
    storage: str = "cuda",
) -> FA3ReplayCacheRegistration:
    """Install the exact replay cache after Grid scope and split2 wrappers."""

    global _RUNTIME
    selected = parse_block_indices(block_indices)
    storage = parse_storage(storage)
    stats = FA3ReplayCacheStats()
    if not selected:
        return FA3ReplayCacheRegistration(
            False, (), None, None, None, None, None, None, stats
        )
    if _RUNTIME is not None:
        raise RuntimeError("H3 FA3 replay cache was installed more than once")
    blocks = list(getattr(transformer, "transformer_blocks", ()))
    if len(blocks) != EXPECTED_BLOCK_COUNT:
        raise RuntimeError(
            f"H3 FA3 replay cache requires {EXPECTED_BLOCK_COUNT} blocks, got {len(blocks)}"
        )
    original_checkpoint = getattr(transformer, "_gradient_checkpointing_func", None)
    if not callable(original_checkpoint):
        raise RuntimeError("H3 FA3 replay cache requires callable checkpoint function")
    if parent_split_registration is None or not bool(
        getattr(parent_split_registration, "enabled", False)
    ):
        raise RuntimeError("H3 FA3 replay cache requires the active FA3 split registration")
    if int(getattr(parent_split_registration, "grad_num_splits", -1)) != 2:
        raise RuntimeError("H3 FA3 replay cache requires grad num_splits=2")

    if kernel_config is None:
        from diffusers.models.attention_dispatch import (
            AttentionBackendName,
            _HUB_KERNELS_REGISTRY,
        )

        kernel_config = _HUB_KERNELS_REGISTRY[AttentionBackendName._FLASH_3_HUB]
    original_kernel = getattr(kernel_config, "kernel_fn", None)
    if not callable(original_kernel):
        raise RuntimeError("H3 FA3 replay cache requires a bound kernel function")
    if getattr(original_kernel, "_h3_fa3_replay_cache_wrapper", False):
        raise RuntimeError("H3 FA3 replay cache kernel wrapper was installed twice")
    raw_forward = raw_forward or getattr(kernel_config, "wrapped_forward_fn", None)
    raw_backward = raw_backward or getattr(kernel_config, "wrapped_backward_fn", None)
    if not callable(raw_forward) or not callable(raw_backward):
        raise RuntimeError("H3 FA3 replay cache requires pinned raw forward/backward ops")

    registration = FA3ReplayCacheRegistration(
        enabled=True,
        block_indices=selected,
        transformer=transformer,
        original_checkpoint=original_checkpoint,
        original_kernel=original_kernel,
        parent_split_registration=parent_split_registration,
        pre_hook=None,
        post_hook=None,
        stats=stats,
        raw_forward_schema=_schema(raw_forward),
        raw_backward_schema=_schema(raw_backward),
        storage=storage,
        cpu_pool=_PinnedPool(stats) if storage == "cpu" else None,
    )
    _RUNTIME = _Runtime(raw_forward, raw_backward, stats)

    @functools.wraps(original_kernel)
    def kernel_wrapper(
        q,
        k,
        v,
        softmax_scale=None,
        causal=False,
        qv=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        window_size=(-1, -1),
        attention_chunk=0,
        softcap=0.0,
        num_splits=1,
        pack_gqa=None,
        deterministic=False,
        sm_margin=0,
        return_attn_probs=False,
    ):
        if not _CACHE_ACTIVE.get():
            return original_kernel(
                q,
                k,
                v,
                softmax_scale=softmax_scale,
                causal=causal,
                qv=qv,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
                window_size=window_size,
                attention_chunk=attention_chunk,
                softcap=softcap,
                num_splits=num_splits,
                pack_gqa=pack_gqa,
                deterministic=deterministic,
                sm_margin=sm_margin,
                return_attn_probs=return_attn_probs,
            )
        errors = []
        if not torch.is_grad_enabled():
            errors.append("grad_disabled")
        if qv is not None or q_descale is not None or k_descale is not None or v_descale is not None:
            errors.append("optional_qv_or_descale")
        if bool(causal) or tuple(window_size) != (-1, -1) or int(attention_chunk) != 0:
            errors.append("mask_or_chunk")
        if float(softcap) != 0.0 or int(num_splits) != 1:
            errors.append("softcap_or_input_splits")
        if pack_gqa is not None or bool(deterministic) or int(sm_margin) != 0:
            errors.append("pack_deterministic_or_margin")
        if bool(return_attn_probs):
            errors.append("return_attn_probs")
        if q.ndim != 4 or tuple(q.shape[:1] + q.shape[2:]) != (
            1,
            EXPECTED_HEADS,
            EXPECTED_HEAD_DIM,
        ):
            errors.append(f"q_shape={tuple(q.shape)}")
        if k.shape != q.shape or v.shape != q.shape:
            errors.append("kv_shape")
        if errors:
            stats.unexpected_kernel_contract_calls += 1
            raise RuntimeError("H3 FA3 replay cache kernel contract failed: " + ",".join(errors))

        # The parent split wrapper is intentionally bypassed for this exact
        # compact implementation, so retain its entry-level physical census.
        parent = parent_split_registration.stats
        parent.total_calls += 1
        parent.grad_calls += 1
        parent.rewritten_calls += 1
        parent.grad_rewritten_calls += 1
        stats.compact_attention_entries += 1
        scale = q.shape[-1] ** (-0.5) if softmax_scale is None else float(softmax_scale)
        return _CompactFA3Autograd.apply(q, k, v, float(scale))

    kernel_wrapper._h3_fa3_replay_cache_wrapper = True
    kernel_config.kernel_fn = kernel_wrapper

    state = {"grad": False, "call_index": 0, "selected": 0}
    cuda_context_fn = functools.partial(
        create_selective_checkpoint_contexts, [_fa3_forward_compact._opoverload]
    )

    def pre_hook(_module: Any, _inputs: tuple[Any, ...]) -> None:
        state["grad"] = bool(torch.is_grad_enabled())
        state["call_index"] = 0
        state["selected"] = 0
        if state["grad"]:
            stats.grad_transformer_forwards += 1

    def post_hook(_module: Any, _inputs: tuple[Any, ...], _output: Any) -> None:
        if not state["grad"]:
            return
        if state["call_index"] != EXPECTED_BLOCK_COUNT or state["selected"] != len(selected):
            stats.unexpected_checkpoint_contract_calls += 1
            raise RuntimeError(
                "H3 FA3 replay cache checkpoint contract failed: "
                f"calls={state['call_index']} selected={state['selected']} "
                f"expected={EXPECTED_BLOCK_COUNT}/{len(selected)}"
            )
        stats.completed_grad_transformer_forwards += 1

    def checkpoint_wrapper(function: Callable[..., Any], *args: Any, **kwargs: Any):
        if not torch.is_grad_enabled():
            return original_checkpoint(function, *args, **kwargs)
        index = int(state["call_index"])
        state["call_index"] = index + 1
        if index not in selected:
            return original_checkpoint(function, *args, **kwargs)
        if "context_fn" in kwargs or "use_reentrant" in kwargs:
            stats.unexpected_checkpoint_contract_calls += 1
            raise RuntimeError(
                "H3 FA3 replay cache observed an existing checkpoint policy override"
            )
        state["selected"] += 1
        stats.selected_checkpoint_wraps += 1
        context_fn = (
            cuda_context_fn
            if registration.storage == "cuda"
            else functools.partial(_create_cpu_cache_contexts, registration, index)
        )

        def scoped(*fn_args: Any, **fn_kwargs: Any):
            stats.selected_scoped_executions += 1
            token = _CACHE_ACTIVE.set(True)
            try:
                return function(*fn_args, **fn_kwargs)
            finally:
                _CACHE_ACTIVE.reset(token)

        # Pinned Diffusers wraps torch.checkpoint in a closure that accepts
        # only ``(module, *args)`` and therefore cannot forward context_fn.
        # Use the exact same non-reentrant policy directly for selected calls.
        # The Grid key wrapper is installed outside this wrapper, so its
        # scoped function remains the callable checkpointed here.
        return checkpoint(
            scoped,
            *args,
            use_reentrant=False,
            context_fn=context_fn,
            **kwargs,
        )

    registration.pre_hook = transformer.register_forward_pre_hook(pre_hook)
    registration.post_hook = transformer.register_forward_hook(post_hook, always_call=True)
    transformer._gradient_checkpointing_func = checkpoint_wrapper
    logger.info(
        "[h3-a100][fa3-replay-cache] installed blocks={} storage={} "
        "cached_outputs=out+lse split=2",
        list(selected),
        storage,
    )
    return registration


def validate_cycle(
    registration: FA3ReplayCacheRegistration, start: dict[str, int]
) -> dict[str, int]:
    if not registration.enabled:
        return {}
    current = registration.snapshot()
    delta = {key: int(current[key]) - int(start.get(key, 0)) for key in current}
    selected_per_cycle = len(registration.block_indices) * EXPECTED_GRAD_TRANSFORMER_FORWARDS
    expected = {
        "grad_transformer_forwards": EXPECTED_GRAD_TRANSFORMER_FORWARDS,
        "completed_grad_transformer_forwards": EXPECTED_GRAD_TRANSFORMER_FORWARDS,
        "selected_checkpoint_wraps": selected_per_cycle,
        "selected_scoped_executions": 2 * selected_per_cycle,
        "compact_attention_entries": 2 * selected_per_cycle,
        "compact_forward_impl_calls": selected_per_cycle,
        "compact_backward_calls": selected_per_cycle,
        "unexpected_kernel_contract_calls": 0,
        "unexpected_checkpoint_contract_calls": 0,
        "unexpected_cpu_storage_calls": 0,
    }
    if registration.storage == "cpu":
        expected.update(
            {
                "cpu_d2h_entries": selected_per_cycle,
                "cpu_h2d_entries": selected_per_cycle,
                "cpu_d2h_tensors": 2 * selected_per_cycle,
                "cpu_h2d_tensors": 2 * selected_per_cycle,
            }
        )
    else:
        expected.update(
            {
                "cpu_d2h_entries": 0,
                "cpu_h2d_entries": 0,
                "cpu_d2h_tensors": 0,
                "cpu_h2d_tensors": 0,
                "cpu_d2h_bytes": 0,
                "cpu_h2d_bytes": 0,
                "cpu_pool_allocated_bytes": 0,
                "cpu_pool_reused_tensors": 0,
                "cpu_pool_busy_misses": 0,
            }
        )
    errors = [
        f"{key}={delta[key]} expected={value}"
        for key, value in expected.items()
        if delta[key] != value
    ]
    if delta["cached_logical_bytes"] <= 0:
        errors.append("cached_logical_bytes=0")
    if registration.storage == "cpu":
        if delta["cpu_d2h_bytes"] != delta["cached_logical_bytes"]:
            errors.append(
                "cpu_d2h_bytes="
                f"{delta['cpu_d2h_bytes']} expected={delta['cached_logical_bytes']}"
            )
        if delta["cpu_h2d_bytes"] != delta["cached_logical_bytes"]:
            errors.append(
                "cpu_h2d_bytes="
                f"{delta['cpu_h2d_bytes']} expected={delta['cached_logical_bytes']}"
            )
    if errors:
        raise RuntimeError("H3 FA3 replay cache cycle failed: " + "; ".join(errors))
    return delta
