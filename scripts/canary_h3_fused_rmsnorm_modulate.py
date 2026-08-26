#!/usr/bin/env python3
"""Production-shape numerical/timing canary for the no-grad fused RMSNorm path."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F

from h3_a100.triton_fused_pointwise import (
    fused_modulate,
    fused_rmsnorm_modulate,
)


HIDDEN_SIZE = 5376
EPS = 1e-5
NORM_L2_MAX = 5e-5
COSINE_MIN = 0.999999
MAX_ABS_MAX = 0.0625


def _reference(value, weight, scale, shift, indices):
    normalized = F.rms_norm(value, (HIDDEN_SIZE,), weight=weight, eps=EPS)
    return fused_modulate(normalized, scale, shift, indices)


def _candidate(value, weight, scale, shift, indices):
    return fused_rmsnorm_modulate(
        value,
        weight,
        scale,
        shift,
        indices,
        eps=EPS,
    )


def _timed(function, *args):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = function(*args)
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)), output


def _metrics(reference, candidate):
    left = reference.float().reshape(-1)
    right = candidate.float().reshape(-1)
    chunk = 4_000_000
    left_sq = right_sq = dot = delta_sq = 0.0
    max_abs = 0.0
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
        "bitwise_equal": bool(torch.equal(reference, candidate)),
        "max_abs": max_abs,
        "normalized_l2": (delta_sq / left_sq) ** 0.5,
        "cosine": dot / (left_sq * right_sq) ** 0.5,
        "finite": bool(torch.isfinite(reference).all() and torch.isfinite(candidate).all()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=37760)
    parser.add_argument("--pairs", type=int, default=10)
    args = parser.parse_args()

    torch.manual_seed(20260826)
    value = torch.randn(
        (1, args.sequence_length, HIDDEN_SIZE), device="cuda", dtype=torch.bfloat16
    )
    weight = (
        torch.randn((HIDDEN_SIZE,), device="cuda", dtype=torch.bfloat16) * 0.02 + 1
    )
    scale = torch.randn((6, HIDDEN_SIZE), device="cuda", dtype=torch.bfloat16) * 0.05
    shift = torch.randn((6, HIDDEN_SIZE), device="cuda", dtype=torch.bfloat16) * 0.05
    indices = torch.arange(args.sequence_length, device="cuda", dtype=torch.int64) % 6

    for _ in range(3):
        _reference(value, weight, scale, shift, indices)
        _candidate(value, weight, scale, shift, indices)
    torch.cuda.synchronize()

    reference_ms = []
    candidate_ms = []
    reference_output = candidate_output = None
    for pair in range(args.pairs):
        order = ("reference", "candidate") if pair % 2 == 0 else ("candidate", "reference")
        for name in order:
            elapsed, output = _timed(
                _reference if name == "reference" else _candidate,
                value,
                weight,
                scale,
                shift,
                indices,
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
    if parity["normalized_l2"] > NORM_L2_MAX:
        errors.append(f"normalized_l2={parity['normalized_l2']} > {NORM_L2_MAX}")
    if parity["cosine"] < COSINE_MIN:
        errors.append(f"cosine={parity['cosine']} < {COSINE_MIN}")
    if parity["max_abs"] > MAX_ABS_MAX:
        errors.append(f"max_abs={parity['max_abs']} > {MAX_ABS_MAX}")
    result = {
        "status": "PASS" if not errors else "FAIL_NUMERICAL_GATE",
        "errors": errors,
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "reference_median_ms": statistics.median(reference_ms),
        "candidate_median_ms": statistics.median(candidate_ms),
        "paired_median_delta_ms": statistics.median(
            candidate - reference
            for reference, candidate in zip(reference_ms, candidate_ms)
        ),
        "reference_ms": reference_ms,
        "candidate_ms": candidate_ms,
        "parity": parity,
        "thresholds": {
            "normalized_l2_max": NORM_L2_MAX,
            "cosine_min": COSINE_MIN,
            "max_abs_max": MAX_ABS_MAX,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

