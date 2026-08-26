#!/usr/bin/env python3
"""Two-rank FSDP2 canary for grad Q/K fusion and live frozen weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.fsdp import fully_shard

from h3_a100.fused_qk_rmsnorm_rotary import (
    FusedQKStats,
    _FusedQKRMSNormRotaryAutograd,
)
from h3_a100.fused_rotary import FusedRotaryStats
from h3_a100.triton_fused_rotary import fused_apply_rotary_emb


HEADS = 56
HEAD_DIM = 128
ROTARY_DIM = 96
EPS = 1e-5


class Unit(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.ones(HEAD_DIM, dtype=torch.bfloat16), requires_grad=False
        )
        self.live_receipt = None
        self.qk_stats = FusedQKStats()
        self.rotary_stats = FusedRotaryStats()

    def forward(self, hidden, cos, sin):
        self.live_receipt = {
            "weight_dtype": str(self.weight.dtype),
            "weight_device": str(self.weight.device),
            "weight_type": type(self.weight).__name__,
            "weight_contiguous": bool(self.weight.is_contiguous()),
            "weight_numel": int(self.weight.numel()),
        }
        return _FusedQKRMSNormRotaryAutograd.apply(
            hidden,
            self.weight,
            cos,
            sin,
            EPS,
            self.qk_stats,
            self.rotary_stats,
        )


def metrics(reference, candidate):
    left = reference.float()
    right = candidate.float()
    delta = left - right
    left_sq = float((left * left).sum(dtype=torch.float64))
    right_sq = float((right * right).sum(dtype=torch.float64))
    delta_sq = float((delta * delta).sum(dtype=torch.float64))
    dot = float((left * right).sum(dtype=torch.float64))
    return {
        "finite": bool(torch.isfinite(reference).all() and torch.isfinite(candidate).all()),
        "max_abs": float(delta.abs().max()),
        "normalized_l2": (delta_sq / left_sq) ** 0.5,
        "cosine": dot / (left_sq * right_sq) ** 0.5,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=128)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(20260826)

    module = fully_shard(Unit().to(device), reshard_after_forward=True)
    base_hidden = torch.randn(
        (1, args.sequence_length, HEADS, HEAD_DIM),
        device=device,
        dtype=torch.bfloat16,
    )
    cos = torch.randn((args.sequence_length, ROTARY_DIM), device=device, dtype=torch.float32)
    sin = torch.randn_like(cos)
    grad_output = torch.randn_like(base_hidden)

    reference_hidden = base_hidden.detach().clone().requires_grad_(True)
    reference = fused_apply_rotary_emb(
        F.rms_norm(
            reference_hidden,
            (HEAD_DIM,),
            weight=torch.ones(HEAD_DIM, device=device, dtype=torch.bfloat16),
            eps=EPS,
        ),
        cos,
        sin,
    )
    reference_gradient = torch.autograd.grad(reference, reference_hidden, grad_output)[0]

    candidate_hidden = base_hidden.detach().clone().requires_grad_(True)
    candidate = module(candidate_hidden, cos, sin)
    candidate_gradient = torch.autograd.grad(candidate, candidate_hidden, grad_output)[0]
    torch.cuda.synchronize(device)

    output_parity = metrics(reference, candidate)
    gradient_parity = metrics(reference_gradient, candidate_gradient)
    output_digest = hashlib.sha256(
        candidate.detach().cpu().contiguous().view(torch.uint8).numpy()
    ).hexdigest()
    gradient_digest = hashlib.sha256(
        candidate_gradient.detach().cpu().contiguous().view(torch.uint8).numpy()
    ).hexdigest()
    local = {
        "rank": rank,
        "local_rank": local_rank,
        "live_weight": module.live_receipt,
        "output_parity": output_parity,
        "gradient_parity": gradient_parity,
        "output_sha256": output_digest,
        "gradient_sha256": gradient_digest,
        "candidate_backward_calls": module.qk_stats.fused_grad_qk_backward_calls,
    }
    rows = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(rows, local)
    errors = []
    for row in rows:
        live = row["live_weight"]
        if live["weight_dtype"] != "torch.bfloat16" or not live["weight_device"].startswith("cuda"):
            errors.append(f"rank{row['rank']} invalid dtype/device {live}")
        if not live["weight_contiguous"] or live["weight_numel"] != HEAD_DIM:
            errors.append(f"rank{row['rank']} invalid live weight {live}")
        for label, values, l2_limit, cosine_limit, max_limit in (
            ("output", row["output_parity"], 5e-5, 0.999999, 0.0625),
            ("gradient", row["gradient_parity"], 5e-4, 0.99999, 0.125),
        ):
            if not values["finite"]:
                errors.append(f"rank{row['rank']} {label} non-finite")
            if values["normalized_l2"] > l2_limit:
                errors.append(f"rank{row['rank']} {label} nL2 {values['normalized_l2']}")
            if values["cosine"] < cosine_limit:
                errors.append(f"rank{row['rank']} {label} cosine {values['cosine']}")
            if values["max_abs"] > max_limit:
                errors.append(f"rank{row['rank']} {label} max_abs {values['max_abs']}")
        if row["candidate_backward_calls"] != 1:
            errors.append(f"rank{row['rank']} backward calls {row['candidate_backward_calls']}")
    if len({row["output_sha256"] for row in rows}) != 1:
        errors.append("candidate output SHA differs across ranks")
    if len({row["gradient_sha256"] for row in rows}) != 1:
        errors.append("candidate gradient SHA differs across ranks")
    result = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "world_size": dist.get_world_size(),
        "torch": torch.__version__,
        "rows": rows,
    }
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    dist.barrier()
    dist.destroy_process_group()
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
