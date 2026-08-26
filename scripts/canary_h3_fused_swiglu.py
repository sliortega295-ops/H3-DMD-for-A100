#!/usr/bin/env python3
"""Bounded A100 exactness/checkpoint/timing canary for fused H3 SwiGLU."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from h3_a100.fused_swiglu import FusedSwiGLUStats, _FusedSwiGLUAutograd
from h3_a100.triton_fused_swiglu import fused_swiglu


def _reference(projected: torch.Tensor) -> torch.Tensor:
    value, gate = projected.chunk(2, dim=-1)
    return value * F.silu(gate)


def _metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | bool]:
    ref = reference.float()
    cand = candidate.float()
    delta = cand - ref
    denom = torch.linalg.vector_norm(ref).clamp_min(1e-12)
    ref_flat = ref.reshape(-1)
    cand_flat = cand.reshape(-1)
    cosine = F.cosine_similarity(ref_flat, cand_flat, dim=0)
    return {
        "bitwise_equal": bool(torch.equal(reference, candidate)),
        "max_abs": float(delta.abs().max().item()),
        "normalized_l2": float((torch.linalg.vector_norm(delta) / denom).item()),
        "cosine": float(cosine.item()),
        "finite": bool(torch.isfinite(cand).all().item()),
    }


def _require_close(label: str, metrics: dict[str, float | bool]) -> None:
    # This gate is deliberately much tighter than the end-to-end H3 parity
    # thresholds.  It permits only the local transcendental implementation's
    # BF16 rounding difference; it does not relax the workload gate.
    errors = []
    if not metrics["finite"]:
        errors.append("non-finite")
    if float(metrics["max_abs"]) > 0.03125:
        errors.append(f"max_abs={metrics['max_abs']}")
    if float(metrics["normalized_l2"]) > 0.002:
        errors.append(f"normalized_l2={metrics['normalized_l2']}")
    if float(metrics["cosine"]) < 0.99999:
        errors.append(f"cosine={metrics['cosine']}")
    if errors:
        raise RuntimeError(f"{label} parity failed: {'; '.join(errors)}")


def _time_cuda(function, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = perf_counter()
    for _ in range(iterations):
        function()
    torch.cuda.synchronize()
    return (perf_counter() - start) * 1000.0 / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--real-sequence-length", type=int, default=0)
    parser.add_argument("--inner-dim", type=int, default=14336)
    parser.add_argument("--grad-sequence-length", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(20260826)

    forward_sequence = args.real_sequence_length or args.sequence_length
    projected = torch.randn(
        (1, forward_sequence, 2 * args.inner_dim), device=device, dtype=torch.bfloat16
    )
    with torch.no_grad():
        reference = _reference(projected)
        candidate = fused_swiglu(projected)
        forward_metrics = _metrics(reference, candidate)
        _require_close("forward", forward_metrics)
        reference_ms = _time_cuda(lambda: _reference(projected), 2, args.iterations)
        candidate_ms = _time_cuda(lambda: fused_swiglu(projected), 2, args.iterations)
    del reference, candidate, projected
    torch.cuda.empty_cache()

    grad_shape = (1, args.grad_sequence_length, 2 * args.inner_dim)
    base = torch.randn(grad_shape, device=device, dtype=torch.bfloat16)
    upstream = torch.randn(
        (1, args.grad_sequence_length, args.inner_dim),
        device=device,
        dtype=torch.bfloat16,
    )
    reference_input = base.detach().clone().requires_grad_(True)
    candidate_input = base.detach().clone().requires_grad_(True)
    reference_output = _reference(reference_input)
    stats = FusedSwiGLUStats()
    candidate_output = _FusedSwiGLUAutograd.apply(candidate_input, stats)
    output_metrics = _metrics(reference_output, candidate_output)
    _require_close("grad-forward", output_metrics)
    reference_output.backward(upstream)
    candidate_output.backward(upstream)
    gradient_metrics = _metrics(reference_input.grad, candidate_input.grad)
    _require_close("input-gradient", gradient_metrics)

    checkpoint_reference_input = base.detach().clone().requires_grad_(True)
    checkpoint_candidate_input = base.detach().clone().requires_grad_(True)
    checkpoint_stats = FusedSwiGLUStats()
    checkpoint_reference = checkpoint(_reference, checkpoint_reference_input, use_reentrant=False)
    checkpoint_candidate = checkpoint(
        lambda value: _FusedSwiGLUAutograd.apply(value, checkpoint_stats),
        checkpoint_candidate_input,
        use_reentrant=False,
    )
    checkpoint_output_metrics = _metrics(checkpoint_reference, checkpoint_candidate)
    _require_close("checkpoint-forward", checkpoint_output_metrics)
    checkpoint_reference.backward(upstream)
    checkpoint_candidate.backward(upstream)
    checkpoint_gradient_metrics = _metrics(
        checkpoint_reference_input.grad, checkpoint_candidate_input.grad
    )
    _require_close("checkpoint-gradient", checkpoint_gradient_metrics)

    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    result = {
        "status": "PASS",
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "sequence_length": forward_sequence,
        "inner_dim": args.inner_dim,
        "grad_sequence_length": args.grad_sequence_length,
        "forward": forward_metrics,
        "grad_forward": output_metrics,
        "input_gradient": gradient_metrics,
        "checkpoint_forward": checkpoint_output_metrics,
        "checkpoint_input_gradient": checkpoint_gradient_metrics,
        "custom_backward_calls": stats.fused_backward_calls,
        "checkpoint_custom_backward_calls": checkpoint_stats.fused_backward_calls,
        "reference_ms": reference_ms,
        "candidate_ms": candidate_ms,
        "speedup": reference_ms / candidate_ms,
        "peak_allocated_gib": peak_allocated / 1024**3,
        "peak_reserved_gib": peak_reserved / 1024**3,
        "driver_free_gib": free_bytes / 1024**3,
        "driver_total_gib": total_bytes / 1024**3,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
