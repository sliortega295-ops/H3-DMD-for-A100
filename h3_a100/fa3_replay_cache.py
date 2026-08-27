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
from collections.abc import Callable, Iterable
from typing import Any

import torch
from loguru import logger
from torch.utils.checkpoint import create_selective_checkpoint_contexts


EXPECTED_BLOCK_COUNT = 50
EXPECTED_GRAD_TRANSFORMER_FORWARDS = 6
EXPECTED_HEADS = 56
EXPECTED_HEAD_DIM = 128
COMPACT_FORWARD_SCHEMA = (
    "h3_a100::fa3_forward_compact(Tensor q, Tensor k, Tensor v, "
    "float softmax_scale) -> (Tensor, Tensor)"
)


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

    def snapshot(self) -> dict[str, int]:
        return dataclasses.asdict(self.stats)

    def receipt(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "block_indices": list(self.block_indices),
            "compact_forward_schema": COMPACT_FORWARD_SCHEMA,
            "raw_forward_schema": self.raw_forward_schema,
            "raw_backward_schema": self.raw_backward_schema,
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
) -> FA3ReplayCacheRegistration:
    """Install the exact replay cache after Grid scope and split2 wrappers."""

    global _RUNTIME
    selected = parse_block_indices(block_indices)
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
        True,
        selected,
        transformer,
        original_checkpoint,
        original_kernel,
        parent_split_registration,
        None,
        None,
        stats,
        _schema(raw_forward),
        _schema(raw_backward),
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
    context_fn = functools.partial(
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
        if "context_fn" in kwargs:
            stats.unexpected_checkpoint_contract_calls += 1
            raise RuntimeError("H3 FA3 replay cache observed an existing checkpoint context_fn")
        state["selected"] += 1
        stats.selected_checkpoint_wraps += 1

        def scoped(*fn_args: Any, **fn_kwargs: Any):
            stats.selected_scoped_executions += 1
            token = _CACHE_ACTIVE.set(True)
            try:
                return function(*fn_args, **fn_kwargs)
            finally:
                _CACHE_ACTIVE.reset(token)

        return original_checkpoint(
            scoped,
            *args,
            context_fn=context_fn,
            **kwargs,
        )

    registration.pre_hook = transformer.register_forward_pre_hook(pre_hook)
    registration.post_hook = transformer.register_forward_hook(post_hook, always_call=True)
    transformer._gradient_checkpointing_func = checkpoint_wrapper
    logger.info(
        "[h3-a100][fa3-replay-cache] installed blocks={} cached_outputs=out+lse split=2",
        list(selected),
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
    }
    errors = [
        f"{key}={delta[key]} expected={value}"
        for key, value in expected.items()
        if delta[key] != value
    ]
    if delta["cached_logical_bytes"] <= 0:
        errors.append("cached_logical_bytes=0")
    if errors:
        raise RuntimeError("H3 FA3 replay cache cycle failed: " + "; ".join(errors))
    return delta
