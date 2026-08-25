"""Opt-in exact Nsight Systems capture around one H3 DMD outer cycle.

Only local rank zero controls the node-level CUDA profiler API.  All ranks
still emit a rank-qualified NVTX range and an immutable receipt.  The helper is
disabled by default and does not change the unprofiled workload path.
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import time
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.distributed as dist


def enabled() -> bool:
    return os.environ.get("H3_NSYS_EXACT_RANGE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _check_cuda_status(result: Any, operation: str) -> None:
    status = result[0] if isinstance(result, tuple) and result else result
    if status is None:
        return
    try:
        code = int(status)
    except (TypeError, ValueError):
        code = int(getattr(status, "value", 0))
    if code != 0:
        raise RuntimeError(f"{operation} failed with CUDA runtime status {code}")


def _write_receipt(value: dict[str, Any]) -> None:
    root = os.environ.get("H3_NSYS_RECEIPT_DIR", "").strip()
    if not root:
        raise RuntimeError("H3_NSYS_RECEIPT_DIR is required for exact profiling")
    path = Path(root) / f"rank_{value['rank']:03d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite Nsight receipt: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


@contextlib.contextmanager
def exact_cycle_range(current_iter: int) -> Iterator[None]:
    """Capture exactly one Student1+Fake5 cycle when profiling is enabled."""
    if not enabled():
        yield
        return
    if current_iter != 0:
        raise RuntimeError("H3 exact Nsight capture is one-cycle-only")
    if not torch.cuda.is_available() or not dist.is_initialized():
        raise RuntimeError("H3 exact Nsight capture requires initialized CUDA world")

    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank < 0:
        raise RuntimeError("LOCAL_RANK is required for exact Nsight capture")
    controls_api = local_rank == 0
    cudart = torch.cuda.cudart()

    # This rendezvous is outside h3/full_cycle.  It prevents peers from issuing
    # cycle kernels before the node-local controller has started collection.
    dist.barrier()
    if controls_api:
        _check_cuda_status(cudart.cudaProfilerStart(), "cudaProfilerStart")
    dist.barrier()
    # A node-level Nsight session needs a short propagation window after the
    # controller process enters cudaProfilerStart.  Without it, one peer in a
    # real 8-process capture can begin its first nested NVTX ranges before that
    # process' collector has switched from armed to active.  This delay and its
    # rendezvous are profiling-only and sit outside h3/full_cycle.
    settle_seconds = float(os.environ.get("H3_NSYS_START_SETTLE_SECONDS", "0.25"))
    if settle_seconds < 0.0 or settle_seconds > 5.0:
        raise RuntimeError(
            "H3_NSYS_START_SETTLE_SECONDS must be in the closed interval [0, 5]"
        )
    if settle_seconds:
        time.sleep(settle_seconds)
    dist.barrier()

    start_unix_ns = time.time_ns()
    start_perf_ns = time.perf_counter_ns()
    label = f"h3/full_cycle/rank={rank}"
    torch.cuda.nvtx.range_push(label)
    error: BaseException | None = None
    try:
        yield
    except BaseException as exc:
        error = exc
        raise
    finally:
        torch.cuda.nvtx.range_pop()
        stop_perf_ns = time.perf_counter_ns()
        stop_unix_ns = time.time_ns()
        if error is None:
            # Close the rank-qualified range first.  The synchronization needed
            # to stop one node-level capture is deliberately outside it.
            dist.barrier()
            if controls_api:
                _check_cuda_status(cudart.cudaProfilerStop(), "cudaProfilerStop")
            dist.barrier()
        elif controls_api:
            # Best-effort finalization without a collective that could deadlock
            # after another rank has already failed.
            _check_cuda_status(cudart.cudaProfilerStop(), "cudaProfilerStop")

        _write_receipt(
            {
                "schema_version": "h3_a100.nsys_exact_cycle.v1",
                "status": "PASS" if error is None else "FAIL",
                "diagnostic_only": True,
                "formal_timing_eligible": False,
                "rank": rank,
                "local_rank": local_rank,
                "controls_profiler_api": controls_api,
                "logical_window": "Student1+Fake5",
                "current_iter": current_iter,
                "start_unix_ns": start_unix_ns,
                "stop_unix_ns": stop_unix_ns,
                "elapsed_ms": (stop_perf_ns - start_perf_ns) / 1_000_000.0,
                "host": socket.gethostname(),
                "error_type": None if error is None else type(error).__name__,
                "error": None if error is None else str(error),
            }
        )


__all__ = ["enabled", "exact_cycle_range"]
