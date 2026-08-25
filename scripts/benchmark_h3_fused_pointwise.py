#!/usr/bin/env python3
"""Bounded A100 canary for the exact no-grad H3 pointwise kernels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from h3_a100.triton_fused_pointwise import fused_modulate, fused_residual


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-length", type=int, default=37760)
    parser.add_argument("--hidden-size", type=int, default=5376)
    parser.add_argument("--table-rows", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--output")
    return parser.parse_args()


@torch.no_grad()
def timed(function, *, warmup: int, iterations: int):
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    values = []
    result = None
    for _ in range(iterations):
        start.record()
        result = function()
        stop.record()
        stop.synchronize()
        values.append(start.elapsed_time(stop))
    return values, result


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)
    device = torch.device("cuda", 0)
    shape = (1, args.sequence_length, args.hidden_size)
    value = torch.randn(shape, device=device, dtype=torch.bfloat16)
    branch = torch.randn_like(value)
    scale = torch.randn(
        (args.table_rows, args.hidden_size), device=device, dtype=torch.bfloat16
    ) * 0.05
    shift = torch.randn_like(scale) * 0.05
    gate = torch.randn_like(scale) * 0.05
    indices = torch.arange(args.sequence_length, device=device, dtype=torch.int64) % args.table_rows

    def reference():
        modulation = value * (1.0 + scale.index_select(0, indices)) + shift.index_select(0, indices)
        residual = value + gate.index_select(0, indices) * branch
        return modulation, residual

    def candidate():
        return (
            fused_modulate(value, scale, shift, indices),
            fused_residual(value, gate, branch, indices),
        )

    reference_ms, reference_output = timed(
        reference, warmup=args.warmup, iterations=args.iterations
    )
    candidate_ms, candidate_output = timed(
        candidate, warmup=args.warmup, iterations=args.iterations
    )
    reference_median = float(torch.tensor(reference_ms).median())
    candidate_median = float(torch.tensor(candidate_ms).median())
    payload = {
        "status": "PASS"
        if all(torch.equal(left, right) for left, right in zip(reference_output, candidate_output))
        else "FAIL",
        "shape": list(shape),
        "dtype": str(value.dtype),
        "device": torch.cuda.get_device_name(device),
        "reference_ms": reference_ms,
        "candidate_ms": candidate_ms,
        "reference_median_ms": reference_median,
        "candidate_median_ms": candidate_median,
        "micro_speedup": reference_median / candidate_median,
        "modulation_bitwise_equal": torch.equal(reference_output[0], candidate_output[0]),
        "residual_bitwise_equal": torch.equal(reference_output[1], candidate_output[1]),
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "claim_boundary": "synthetic operator canary; not an end-to-end speed claim",
    }
    if payload["status"] != "PASS":
        raise RuntimeError(json.dumps(payload, indent=2))
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
        output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
