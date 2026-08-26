#!/usr/bin/env python3
"""Production-shape parity/timing canary for grad Q/K fusion."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F

from h3_a100.fused_qk_rmsnorm_rotary import (
    FusedQKStats,
    _FusedQKRMSNormRotaryAutograd,
)
from h3_a100.fused_rotary import FusedRotaryStats
from h3_a100.triton_fused_rotary import fused_apply_rotary_emb


EPS = 1e-5
HEADS = 56
HEAD_DIM = 128
ROTARY_DIM = 96


def metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | bool]:
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


def timed(function):
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    output, gradient = function()
    stop.record()
    stop.synchronize()
    return float(start.elapsed_time(stop)), output.detach(), gradient.detach()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=37760)
    parser.add_argument("--pairs", type=int, default=5)
    args = parser.parse_args()

    torch.manual_seed(20260826)
    hidden = torch.randn(
        (1, args.sequence_length, HEADS, HEAD_DIM),
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    weight = (
        torch.randn((HEAD_DIM,), device="cuda", dtype=torch.bfloat16) * 0.02 + 1
    )
    weight.requires_grad_(False)
    cos = torch.randn(
        (args.sequence_length, ROTARY_DIM), device="cuda", dtype=torch.float32
    )
    sin = torch.randn_like(cos)
    grad_output = torch.randn_like(hidden)
    qk_stats = FusedQKStats()
    rotary_stats = FusedRotaryStats()

    def reference():
        output = fused_apply_rotary_emb(
            F.rms_norm(hidden, (HEAD_DIM,), weight=weight, eps=EPS), cos, sin
        )
        gradient = torch.autograd.grad(output, hidden, grad_output, retain_graph=False)[0]
        return output, gradient

    def candidate():
        output = _FusedQKRMSNormRotaryAutograd.apply(
            hidden, weight, cos, sin, EPS, qk_stats, rotary_stats
        )
        gradient = torch.autograd.grad(output, hidden, grad_output, retain_graph=False)[0]
        return output, gradient

    for _ in range(2):
        reference()
        candidate()
    torch.cuda.synchronize()

    reference_ms: list[float] = []
    candidate_ms: list[float] = []
    reference_output = candidate_output = None
    reference_gradient = candidate_gradient = None
    for pair in range(args.pairs):
        order = ("reference", "candidate") if pair % 2 == 0 else ("candidate", "reference")
        for name in order:
            elapsed, output, gradient = timed(reference if name == "reference" else candidate)
            if name == "reference":
                reference_ms.append(elapsed)
                reference_output = output
                reference_gradient = gradient
            else:
                candidate_ms.append(elapsed)
                candidate_output = output
                candidate_gradient = gradient

    output_parity = metrics(reference_output, candidate_output)
    gradient_parity = metrics(reference_gradient, candidate_gradient)
    errors = []
    gates = (
        ("output", output_parity, 5e-5, 0.999999, 0.0625),
        ("gradient", gradient_parity, 5e-4, 0.99999, 0.125),
    )
    for label, values, l2_limit, cosine_limit, max_limit in gates:
        if not values["finite"]:
            errors.append(f"{label} non-finite")
        if values["normalized_l2"] > l2_limit:
            errors.append(f"{label} normalized_l2={values['normalized_l2']} > {l2_limit}")
        if values["cosine"] < cosine_limit:
            errors.append(f"{label} cosine={values['cosine']} < {cosine_limit}")
        if values["max_abs"] > max_limit:
            errors.append(f"{label} max_abs={values['max_abs']} > {max_limit}")

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
        "output_parity": output_parity,
        "gradient_parity": gradient_parity,
        "candidate_backward_calls": qk_stats.fused_grad_qk_backward_calls,
        "parent_rotary_backward_calls": rotary_stats.fused_grad_backward_calls,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
