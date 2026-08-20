"""PyTorch-2.10 non-reentrant replay audit fix for Grid-1000.

Non-reentrant checkpoint may terminate recomputation through an internal
early-stop exception, so a replay execution must be counted/audited from entry
and ``finally``, not after a normal user-function return.
"""

from __future__ import annotations

from typing import Any, Callable

from loguru import logger

from . import grid_adaln as grid


def install_grid_checkpoint_replay_scope(
    transformer: Any,
    controller: grid.GridAdaLNController,
    *,
    expected_block_count: int = 50,
) -> grid.GridReplayRegistration:
    original = getattr(transformer, "_gradient_checkpointing_func", None)
    if not callable(original):
        raise RuntimeError("Grid replay requires callable native checkpoint function")
    stats = {
        "checkpoint_wrap_count": 0,
        "scoped_execution_count": 0,
        "cache_hit_count": 0,
        "cache_miss_count": 0,
        "missing_key_count": 0,
    }

    def wrapped(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        active = controller.current_scope()
        key = active.key
        if key is None or not controller.has_complete_key(key, expected_block_count):
            stats["missing_key_count"] += 1
            raise RuntimeError("Grid checkpoint created without a staged AdaLN table key")
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
                if miss_delta != 0 or hit_delta != 1:
                    raise RuntimeError(
                        f"Grid checkpoint replay lookup mismatch key={key!r} "
                        f"hits={hit_delta} misses={miss_delta}"
                    )

        return original(scoped_function, *args, **kwargs)

    transformer._gradient_checkpointing_func = wrapped
    logger.info("[h3-a100][grid-adaln] installed early-stop-safe checkpoint replay scope")
    return grid.GridReplayRegistration(transformer, controller, original, stats)


# trainer_setup in grid_adaln resolves this global when it actually runs, so
# replacing it here is sufficient as long as this module is imported before
# trainer construction.
grid.install_grid_checkpoint_replay_scope = install_grid_checkpoint_replay_scope
