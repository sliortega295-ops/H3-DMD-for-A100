"""Exact AdaLN-cache scope restoration for native checkpoint replay.

Iteration 308 showed a ~522 MB/block backward residency staircase.  The shared
model precomputes exact AdaLN modulation before every grad-enabled DiT forward,
but PyTorch's native non-reentrant checkpoint does not preserve our ContextVar
scope when it later replays a block.  Without the scope, the parameter-free
AdaLN handle falls back to its ~520 MB projection FSDP unit and materializes one
projection per replayed block.

This module wraps the *already installed* native checkpoint function.  It
captures the exact AdaLN cache key at checkpoint creation time and restores it
around both the original block execution and backward replay.  It fails closed
if the key is missing, incomplete, or if executing a block causes any AdaLN
cache miss.  Continuous DMD sigma, model math, application forward counts,
FSDP placement and checkpoint boundaries are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from loguru import logger

from .adaln_cache import AdaLNCacheController


@dataclass
class ExactAdaLNReplayRegistration:
    transformer: Any
    controller: AdaLNCacheController
    original_checkpoint: Callable[..., Any]
    stats: dict[str, int]
    closed: bool = False

    def receipt(self) -> dict[str, Any]:
        return {
            "policy": "exact_adaln_checkpoint_replay_scope",
            "stats": dict(self.stats),
            "closed": self.closed,
        }

    def remove(self) -> None:
        if self.closed:
            return
        self.transformer._gradient_checkpointing_func = self.original_checkpoint
        self.closed = True


def install_exact_adaln_checkpoint_replay_scope(
    transformer: Any,
    controller: AdaLNCacheController,
    *,
    expected_block_count: int = 50,
) -> ExactAdaLNReplayRegistration:
    """Restore the precomputed AdaLN key whenever checkpoint executes/replays.

    Install this *after* checkpoint-boundary CPU staging, so the wrapper chain
    is:

        exact AdaLN scope -> boundary CPU staging -> native torch checkpoint

    The nested scoped function is what torch checkpoint stores and replays.
    PyTorch non-reentrant checkpoint may terminate replay through its internal
    early-stop exception before the user function returns, so replay auditing
    is performed from ``finally`` rather than after a normal return.
    """

    original = getattr(transformer, "_gradient_checkpointing_func", None)
    if not callable(original):
        raise RuntimeError("MiniMax transformer has no callable checkpoint function")
    if not bool(getattr(transformer, "gradient_checkpointing", False)):
        raise RuntimeError("exact AdaLN replay scope requires gradient checkpointing")

    stats = {
        "checkpoint_wrap_count": 0,
        "scoped_execution_count": 0,
        "captured_key_missing_count": 0,
        "captured_key_incomplete_count": 0,
        "cache_hit_count": 0,
        "cache_miss_count": 0,
        "unexpected_hit_delta_count": 0,
    }

    def wrapped(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        active = controller.current_scope()
        key = active.key
        if key is None:
            stats["captured_key_missing_count"] += 1
            raise RuntimeError(
                "Exact AdaLN checkpoint replay has no active cache key. "
                "Refusing to fall back to the giant AdaLN projection bank."
            )
        if not controller.has_complete_key(key, expected_block_count):
            stats["captured_key_incomplete_count"] += 1
            raise RuntimeError(
                f"Exact AdaLN cache key {key!r} is incomplete before checkpoint creation"
            )
        persistent = bool(active.persistent)
        stats["checkpoint_wrap_count"] += 1

        def scoped_function(*fn_args: Any, **fn_kwargs: Any) -> Any:
            before = controller.stats()
            stats["scoped_execution_count"] += 1
            try:
                with controller.scope(key, persistent=persistent):
                    return function(*fn_args, **fn_kwargs)
            finally:
                after = controller.stats()
                hit_delta = int(after.hits - before.hits)
                miss_delta = int(after.misses - before.misses)
                stats["cache_hit_count"] += hit_delta
                stats["cache_miss_count"] += miss_delta
                # AdaLN is the first block-level conditioning operation, so
                # even non-reentrant early-stop replay must have consumed the
                # one cached modulation entry before checkpoint stops.
                if miss_delta != 0:
                    raise RuntimeError(
                        f"Exact AdaLN replay cache miss for key={key!r}; "
                        f"miss_delta={miss_delta}. Projection fallback is forbidden."
                    )
                if hit_delta != 1:
                    stats["unexpected_hit_delta_count"] += 1
                    raise RuntimeError(
                        f"Exact AdaLN replay expected one cache hit per block, got {hit_delta} "
                        f"for key={key!r}"
                    )

        return original(scoped_function, *args, **kwargs)

    transformer._gradient_checkpointing_func = wrapped
    registration = ExactAdaLNReplayRegistration(
        transformer=transformer,
        controller=controller,
        original_checkpoint=original,
        stats=stats,
    )
    logger.info(
        "[h3-a100][exact-adaln-replay] installed expected_blocks={} wrapper_chain=scope->boundary->checkpoint",
        expected_block_count,
    )
    return registration


def install_trainer_patch() -> None:
    """Patch the experiment trainer without perturbing the shared base branch."""

    from .trainer import MiniMaxH3A100DmdTrainer

    if getattr(MiniMaxH3A100DmdTrainer, "_exact_adaln_replay_patch_installed", False):
        return

    original_setup = MiniMaxH3A100DmdTrainer.setup
    original_begin = MiniMaxH3A100DmdTrainer._begin_boundary_offload_cycle
    original_validate = MiniMaxH3A100DmdTrainer._validate_boundary_offload_cycle

    def setup(self, *args: Any, **kwargs: Any):
        result = original_setup(self, *args, **kwargs)
        if not getattr(self, "adaln_cache_enabled", False):
            raise RuntimeError("Exact replay branch requires exact AdaLN caching enabled")
        if int(getattr(self, "activation_checkpoint_segment_size", -1)) != 1:
            raise RuntimeError("Exact replay branch requires native per-block checkpoint segment=1")
        if getattr(self, "boundary_offload_registration", None) is None:
            raise RuntimeError("Exact replay branch requires checkpoint_boundary_cpu to be installed first")
        self.exact_adaln_replay_registration = install_exact_adaln_checkpoint_replay_scope(
            self.shared_model.denoiser_module(),
            self.shared_model.adaln_cache(),
            expected_block_count=50,
        )
        return result

    def begin_cycle(self) -> None:
        original_begin(self)
        registration = getattr(self, "exact_adaln_replay_registration", None)
        self._exact_adaln_replay_cycle_start = (
            None if registration is None else dict(registration.stats)
        )

    def validate_cycle(self, current_iter: int) -> None:
        original_validate(self, current_iter)
        registration = getattr(self, "exact_adaln_replay_registration", None)
        if registration is None:
            raise RuntimeError("exact AdaLN replay registration is missing")
        start = self._exact_adaln_replay_cycle_start
        if start is None:
            raise RuntimeError("exact AdaLN replay cycle baseline is missing")

        def delta(key: str) -> int:
            return int(registration.stats[key]) - int(start.get(key, 0))

        # 1 Student grad DiT + 5 Fake grad DiTs, 50 checkpointed H3 blocks.
        # Each checkpointed block is entered once in original forward and once
        # during non-reentrant replay, even if replay exits through early-stop.
        expected = {
            "checkpoint_wrap_count": 300,
            "scoped_execution_count": 600,
            "cache_hit_count": 600,
            "cache_miss_count": 0,
            "captured_key_missing_count": 0,
            "captured_key_incomplete_count": 0,
            "unexpected_hit_delta_count": 0,
        }
        observed = {key: delta(key) for key in expected}
        errors = [
            f"{key}={observed[key]} expected={value}"
            for key, value in expected.items()
            if observed[key] != value
        ]
        if errors:
            raise RuntimeError(
                "exact AdaLN checkpoint-replay contract failed at outer iteration "
                f"{current_iter}: {'; '.join(errors)}"
            )
        logger.info(
            "[h3-a100][exact-adaln-replay] iter={} observed={}",
            current_iter,
            observed,
        )

    MiniMaxH3A100DmdTrainer.setup = setup
    MiniMaxH3A100DmdTrainer._begin_boundary_offload_cycle = begin_cycle
    MiniMaxH3A100DmdTrainer._validate_boundary_offload_cycle = validate_cycle
    MiniMaxH3A100DmdTrainer._exact_adaln_replay_patch_installed = True


install_trainer_patch()
