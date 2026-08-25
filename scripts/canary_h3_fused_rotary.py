#!/usr/bin/env python3
"""Bitwise and timing canary for the pinned MiniMax-H3 rotary helper."""

from __future__ import annotations

import argparse
import json
import time

import torch

from h3_a100.fused_rotary import install_fused_rotary
from h3_a100.triton_fused_rotary import fused_apply_rotary_emb


def _time_cuda(function, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        function()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-length", type=int, default=37760)
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--rotary-dim", type=int, default=96)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    import diffusers.models.transformers.transformer_minimax_h3 as module

    torch.manual_seed(20260825)
    device = torch.device("cuda:0")
    hidden = torch.randn(
        (1, args.sequence_length, args.heads, args.head_dim),
        device=device,
        dtype=torch.bfloat16,
    )
    cos = torch.randn((args.sequence_length, args.rotary_dim), device=device, dtype=torch.float32)
    sin = torch.randn((args.sequence_length, args.rotary_dim), device=device, dtype=torch.float32)
    original = module._apply_rotary_emb

    with torch.no_grad():
        expected = original(hidden, cos, sin)
        observed = fused_apply_rotary_emb(hidden, cos, sin)
    if not torch.equal(expected, observed):
        difference = (expected.float() - observed.float()).abs()
        raise RuntimeError(
            "fused rotary output is not bitwise equal: "
            f"max_abs={difference.max().item()} mismatched={(expected != observed).sum().item()}"
        )

    registration = install_fused_rotary(enabled=True)
    with torch.no_grad():
        wrapped = module._apply_rotary_emb(hidden, cos, sin)
    if not torch.equal(expected, wrapped):
        raise RuntimeError("installed no-grad rotary wrapper changed output")
    grad_input = hidden.detach().clone().requires_grad_(True)
    grad_output = module._apply_rotary_emb(grad_input, cos, sin)
    reference_input = hidden.detach().clone().requires_grad_(True)
    reference_output = original(reference_input, cos, sin)
    if not torch.equal(grad_output, reference_output):
        raise RuntimeError("installed grad rotary wrapper changed output")
    grad_output.float().sum().backward()
    reference_output.float().sum().backward()
    if not torch.equal(grad_input.grad, reference_input.grad):
        raise RuntimeError("installed grad rotary wrapper changed input gradient")

    with torch.no_grad():
        reference_ms = _time_cuda(lambda: original(hidden, cos, sin), args.warmup, args.iterations)
        fused_ms = _time_cuda(
            lambda: fused_apply_rotary_emb(hidden, cos, sin),
            args.warmup,
            args.iterations,
        )
    print(
        json.dumps(
            {
                "status": "PASS",
                "shape": list(hidden.shape),
                "cos_dtype": str(cos.dtype),
                "output_bitwise_equal": True,
                "grad_output_bitwise_equal": True,
                "grad_input_bitwise_equal": True,
                "reference_ms": reference_ms,
                "fused_ms": fused_ms,
                "speedup": reference_ms / fused_ms,
                "registration": registration.receipt(),
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
