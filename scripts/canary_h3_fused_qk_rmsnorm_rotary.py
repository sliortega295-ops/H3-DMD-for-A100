#!/usr/bin/env python3
"""Production-shape parity/timing canary for no-grad Q/K fusion."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F

from h3_a100.triton_fused_rotary import fused_qk_rmsnorm_rotary


EPS = 1e-5
HEADS = 56
HEAD_DIM = 128
ROTARY_DIM = 96


def _metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | bool]:
    left = reference.float().reshape(-1)
    right = candidate.float().reshape(-1)
    left_sq = right_sq = dot = delta_sq = 0.0
    max_abs = 0.0
    chunk = 4_000_000
    for offset in range(0, left.numel(), chunk):
        x = left[offset : offset + chunk]
        y = right[offset : offset + chunk]
        delta = x - y
        left_sq += float((x * x).sum(dtype=torch.float64))
        right_sq += float((y * y).sum(dtype=torch.float64))
        dot += float((x * y).sum(dtype=torch.float64))
        delta_sq += float((delta * delta).sum(dtype=torch.float64))
        max_abs = max(max_abs, float(delta.abs().max()))
    return {
        "finite": bool(torch.isfinite(reference).all() and torch.isfinite(candidate).all()),
        "max_abs": max_abs,
        "normalized_l2": (delta_sq / left_sq) ** 0.5,
        "cosine": dot / (left_sq * right_sq) ** 0.5,
    }


def _timed(function, *args):
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    output = function(*args)
    stop.record()
    stop.synchronize()
    return float(start.elapsed_time(stop)), output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=37760)
    parser.add_argument("--pairs", type=int, default=5)
    args = parser.parse_args()

    import diffusers.models.transformers.transformer_minimax_h3 as module

    torch.manual_seed(20260826)
    hidden = torch.randn(
        (1, args.sequence_length, HEADS, HEAD_DIM),
        device="cuda",
        dtype=torch.bfloat16,
    )
    weight = torch.randn((HEAD_DIM,), device="cuda", dtype=torch.bfloat16) * 0.02 + 1
    cos = torch.randn((args.sequence_length, ROTARY_DIM), device="cuda", dtype=torch.float32)
    sin = torch.randn_like(cos)

    def reference(value, affine, cosine, sine):
        normalized = F.rms_norm(value, (HEAD_DIM,), weight=affine, eps=EPS)
        return module._apply_rotary_emb(normalized, cosine, sine)

    def candidate(value, affine, cosine, sine):
        return fused_qk_rmsnorm_rotary(
            value, affine, cosine, sine, eps=EPS
        )

    for _ in range(2):
        reference(hidden, weight, cos, sin)
        candidate(hidden, weight, cos, sin)
    torch.cuda.synchronize()

    reference_ms: list[float] = []
    candidate_ms: list[float] = []
    reference_output = candidate_output = None
    for pair in range(args.pairs):
        order = ("reference", "candidate") if pair % 2 == 0 else ("candidate", "reference")
        for name in order:
            elapsed, output = _timed(
                reference if name == "reference" else candidate,
                hidden,
                weight,
                cos,
                sin,
            )
            if name == "reference":
                reference_ms.append(elapsed)
                reference_output = output
            else:
                candidate_ms.append(elapsed)
                candidate_output = output

    parity = _metrics(reference_output, candidate_output)
    errors = []
    if not parity["finite"]:
        errors.append("non-finite output")
    if parity["normalized_l2"] > 5e-5:
        errors.append(f"normalized_l2={parity['normalized_l2']} > 5e-5")
    if parity["cosine"] < 0.999999:
        errors.append(f"cosine={parity['cosine']} < 0.999999")
    if parity["max_abs"] > 0.0625:
        errors.append(f"max_abs={parity['max_abs']} > 0.0625")
    result = {
        "status": "PASS" if not errors else "FAIL_NUMERICAL_GATE",
        "errors": errors,
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "shape": list(hidden.shape),
        "reference_median_ms": statistics.median(reference_ms),
        "candidate_median_ms": statistics.median(candidate_ms),
        "paired_median_delta_ms": statistics.median(
            candidate_time - reference_time
            for reference_time, candidate_time in zip(reference_ms, candidate_ms)
        ),
        "reference_ms": reference_ms,
        "candidate_ms": candidate_ms,
        "parity": parity,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
