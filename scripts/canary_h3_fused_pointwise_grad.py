#!/usr/bin/env python3
"""Real-shape exactness and checkpoint canary for grad pointwise fusion."""

from __future__ import annotations

import argparse
import json
import time

import torch
from torch.utils.checkpoint import checkpoint

from h3_a100.fused_block_pointwise import (
    FusedPointwiseStats,
    _FusedModulateAutograd,
    _FusedResidualAutograd,
)


def _reference_modulate(value, scale, shift, indices):
    return value * (1 + scale.index_select(0, indices)) + shift.index_select(0, indices)


def _reference_residual(residual, gate, branch, indices):
    return residual + gate.index_select(0, indices) * branch


def _fused_modulate(value, scale, shift, indices, stats):
    return _FusedModulateAutograd.apply(value, scale, shift, indices, stats)


def _fused_residual(residual, gate, branch, indices, stats):
    return _FusedResidualAutograd.apply(residual, gate, branch, indices, stats)


def _require_equal(name, expected, observed):
    if not torch.equal(expected, observed):
        diff = (expected.float() - observed.float()).abs()
        raise RuntimeError(
            f"{name} is not bitwise equal max_abs={diff.max().item()} "
            f"mismatched={(expected != observed).sum().item()}"
        )


def _one_shape(sequence_length: int, hidden_size: int) -> dict:
    generator = torch.Generator(device="cuda").manual_seed(20260825 + sequence_length)
    shape = (1, sequence_length, hidden_size)
    indices = torch.arange(sequence_length, device="cuda", dtype=torch.int64) % 6
    scale = torch.randn((6, hidden_size), device="cuda", dtype=torch.bfloat16, generator=generator) * 0.05
    shift = torch.randn((6, hidden_size), device="cuda", dtype=torch.bfloat16, generator=generator) * 0.05
    gate = torch.randn((6, hidden_size), device="cuda", dtype=torch.bfloat16, generator=generator) * 0.05
    value = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)
    residual = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)
    branch = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)
    upstream = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)

    ref_value = value.detach().clone().requires_grad_(True)
    fused_value = value.detach().clone().requires_grad_(True)
    stats = FusedPointwiseStats()
    ref_mod = _reference_modulate(ref_value, scale, shift, indices)
    fused_mod = _fused_modulate(fused_value, scale, shift, indices, stats)
    _require_equal("modulate output", ref_mod, fused_mod)
    ref_mod.backward(upstream, retain_graph=False)
    fused_mod.backward(upstream, retain_graph=False)
    _require_equal("modulate input gradient", ref_value.grad, fused_value.grad)

    ref_residual = residual.detach().clone().requires_grad_(True)
    ref_branch = branch.detach().clone().requires_grad_(True)
    fused_residual = residual.detach().clone().requires_grad_(True)
    fused_branch = branch.detach().clone().requires_grad_(True)
    ref_out = _reference_residual(ref_residual, gate, ref_branch, indices)
    fused_out = _fused_residual(fused_residual, gate, fused_branch, indices, stats)
    _require_equal("residual output", ref_out, fused_out)
    ref_out.backward(upstream, retain_graph=False)
    fused_out.backward(upstream, retain_graph=False)
    _require_equal("residual edge gradient", ref_residual.grad, fused_residual.grad)
    _require_equal("branch gradient", ref_branch.grad, fused_branch.grad)

    torch.cuda.synchronize()
    return {
        "sequence_length": sequence_length,
        "hidden_size": hidden_size,
        "output_bitwise": True,
        "input_gradient_bitwise": True,
        "branch_gradient_bitwise": True,
        "stats": stats.__dict__,
    }


def _checkpoint_canary() -> dict:
    torch.manual_seed(20260825)
    sequence_length, hidden_size = 257, 128
    indices = torch.arange(sequence_length, device="cuda", dtype=torch.int64) % 6
    scale = torch.randn((6, hidden_size), device="cuda", dtype=torch.bfloat16) * 0.05
    shift = torch.randn((6, hidden_size), device="cuda", dtype=torch.bfloat16) * 0.05
    gate = torch.randn((6, hidden_size), device="cuda", dtype=torch.bfloat16) * 0.05
    upstream = torch.randn((1, sequence_length, hidden_size), device="cuda", dtype=torch.bfloat16)
    base_x = torch.randn_like(upstream)
    base_branch = torch.randn_like(upstream)
    stats = FusedPointwiseStats()

    ref_x = base_x.detach().clone().requires_grad_(True)
    ref_branch = base_branch.detach().clone().requires_grad_(True)
    fused_x = base_x.detach().clone().requires_grad_(True)
    fused_branch = base_branch.detach().clone().requires_grad_(True)

    def reference_fn(x, branch):
        return _reference_residual(
            x,
            gate,
            _reference_modulate(branch, scale, shift, indices),
            indices,
        )

    def fused_fn(x, branch):
        return _fused_residual(
            x,
            gate,
            _fused_modulate(branch, scale, shift, indices, stats),
            indices,
            stats,
        )

    ref_out = checkpoint(reference_fn, ref_x, ref_branch, use_reentrant=False)
    fused_out = checkpoint(fused_fn, fused_x, fused_branch, use_reentrant=False)
    _require_equal("checkpoint output", ref_out, fused_out)
    ref_out.backward(upstream)
    fused_out.backward(upstream)
    _require_equal("checkpoint residual gradient", ref_x.grad, fused_x.grad)
    _require_equal("checkpoint branch gradient", ref_branch.grad, fused_branch.grad)
    return {
        "output_bitwise": True,
        "residual_gradient_bitwise": True,
        "branch_gradient_bitwise": True,
        "stats": stats.__dict__,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-length", type=int, action="append", default=[])
    parser.add_argument("--hidden-size", type=int, default=5376)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    lengths = args.sequence_length or [37760, 37725]
    started = time.perf_counter()
    shapes = [_one_shape(length, args.hidden_size) for length in lengths]
    checkpoint_result = _checkpoint_canary()
    print(
        json.dumps(
            {
                "status": "PASS",
                "device": torch.cuda.get_device_name(),
                "shapes": shapes,
                "checkpoint": checkpoint_result,
                "wall_seconds": time.perf_counter() - started,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
