"""Opt-in no-grad FA3 split scheduling for the pinned Hub kernel."""

from __future__ import annotations

import dataclasses
import functools
from typing import Any, Callable

import torch


# The controlled cycle contains 25 no-grad DiT calls and 12 grad/replay DiT
# executions. MiniMax-H3 has 50 main plus 2 refiner attention layers.
EXPECTED_NOGRAD_CALLS_PER_CYCLE = 25 * 52
EXPECTED_GRAD_CALLS_PER_CYCLE = 12 * 52


@dataclasses.dataclass
class FA3NoGradSplitStats:
    total_calls: int = 0
    no_grad_calls: int = 0
    grad_calls: int = 0
    rewritten_calls: int = 0
    unexpected_input_num_splits: int = 0


@dataclasses.dataclass
class FA3NoGradSplitRegistration:
    enabled: bool
    num_splits: int
    original_module: str | None
    original_qualname: str | None
    stats: FA3NoGradSplitStats

    def snapshot(self) -> dict[str, int]:
        return dataclasses.asdict(self.stats)

    def receipt(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "num_splits": self.num_splits,
            "original_module": self.original_module,
            "original_qualname": self.original_qualname,
            "stats": self.snapshot(),
        }


def install_fa3_nograd_splits(
    *,
    num_splits: int,
    kernel_config: Any | None = None,
) -> FA3NoGradSplitRegistration:
    """Rewrite only no-grad calls from FA3 num_splits=1 to the candidate.

    The Diffusers backend, model processor, Q/K/V, mask, scale, and grad path
    remain unchanged. Passing kernel_config is supported only to make the
    wrapper independently testable without importing Diffusers.
    """

    num_splits = int(num_splits)
    if num_splits < 1:
        raise ValueError(f"H3 FA3 no-grad num_splits must be >=1, got {num_splits}")
    if num_splits == 1:
        return FA3NoGradSplitRegistration(
            False, 1, None, None, FA3NoGradSplitStats()
        )
    if num_splits != 2:
        raise ValueError(
            "The bounded H3 FA3 candidate only authorizes num_splits=2; "
            f"got {num_splits}"
        )

    if kernel_config is None:
        from diffusers.models.attention_dispatch import (
            AttentionBackendName,
            _HUB_KERNELS_REGISTRY,
        )

        kernel_config = _HUB_KERNELS_REGISTRY[AttentionBackendName._FLASH_3_HUB]
    original: Callable[..., Any] | None = getattr(kernel_config, "kernel_fn", None)
    if original is None:
        raise RuntimeError(
            "H3 FA3 no-grad split candidate requires the pinned local Hub "
            "kernel to be bound before installation"
        )
    if getattr(original, "_h3_fa3_nograd_split_wrapper", False):
        raise RuntimeError("H3 FA3 no-grad split wrapper was installed more than once")

    stats = FA3NoGradSplitStats()

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        stats.total_calls += 1
        if torch.is_grad_enabled():
            stats.grad_calls += 1
            return original(*args, **kwargs)
        stats.no_grad_calls += 1
        observed = int(kwargs.get("num_splits", 1))
        if observed != 1:
            stats.unexpected_input_num_splits += 1
            raise RuntimeError(
                "H3 FA3 no-grad split expected Diffusers num_splits=1, "
                f"observed {observed}"
            )
        kwargs["num_splits"] = num_splits
        stats.rewritten_calls += 1
        return original(*args, **kwargs)

    wrapped._h3_fa3_nograd_split_wrapper = True
    kernel_config.kernel_fn = wrapped
    return FA3NoGradSplitRegistration(
        True,
        num_splits,
        getattr(original, "__module__", None),
        getattr(original, "__qualname__", None),
        stats,
    )


def validate_cycle(
    registration: FA3NoGradSplitRegistration, start: dict[str, int]
) -> dict[str, int]:
    if not registration.enabled:
        return {}
    current = registration.snapshot()
    delta = {key: int(current[key]) - int(start.get(key, 0)) for key in current}
    expected = {
        "total_calls": EXPECTED_NOGRAD_CALLS_PER_CYCLE + EXPECTED_GRAD_CALLS_PER_CYCLE,
        "no_grad_calls": EXPECTED_NOGRAD_CALLS_PER_CYCLE,
        "grad_calls": EXPECTED_GRAD_CALLS_PER_CYCLE,
        "rewritten_calls": EXPECTED_NOGRAD_CALLS_PER_CYCLE,
        "unexpected_input_num_splits": 0,
    }
    errors = [
        f"{key}={delta[key]} expected={value}"
        for key, value in expected.items()
        if delta[key] != value
    ]
    if errors:
        raise RuntimeError(f"H3 FA3 no-grad split cycle contract failed: {'; '.join(errors)}")
    return delta
