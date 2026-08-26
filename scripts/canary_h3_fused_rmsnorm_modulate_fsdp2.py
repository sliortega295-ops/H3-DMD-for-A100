#!/usr/bin/env python3
"""Two-rank FSDP2 canary for live RMSNorm weight materialization."""

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

from h3_a100.triton_fused_pointwise import fused_modulate, fused_rmsnorm_modulate


HIDDEN_SIZE = 5376
EPS = 1e-5


class _Unit(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.ones(HIDDEN_SIZE, dtype=torch.bfloat16), requires_grad=False
        )
        self.live_receipt = None

    def forward(self, value, scale, shift, indices):
        self.live_receipt = {
            "weight_dtype": str(self.weight.dtype),
            "weight_device": str(self.weight.device),
            "weight_type": type(self.weight).__name__,
            "weight_contiguous": bool(self.weight.is_contiguous()),
            "weight_numel": int(self.weight.numel()),
        }
        reference = fused_modulate(
            F.rms_norm(value, (HIDDEN_SIZE,), weight=self.weight, eps=EPS),
            scale,
            shift,
            indices,
        )
        candidate = fused_rmsnorm_modulate(
            value,
            self.weight,
            scale,
            shift,
            indices,
            eps=EPS,
        )
        return reference, candidate


def _metrics(reference, candidate):
    left = reference.float()
    right = candidate.float()
    delta = left - right
    left_sq = float((left * left).sum(dtype=torch.float64))
    right_sq = float((right * right).sum(dtype=torch.float64))
    delta_sq = float((delta * delta).sum(dtype=torch.float64))
    dot = float((left * right).sum(dtype=torch.float64))
    return {
        "max_abs": float(delta.abs().max()),
        "normalized_l2": (delta_sq / left_sq) ** 0.5,
        "cosine": dot / (left_sq * right_sq) ** 0.5,
        "finite": bool(torch.isfinite(reference).all() and torch.isfinite(candidate).all()),
    }


def main():
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

    module = _Unit().to(device)
    module = fully_shard(module, reshard_after_forward=True)
    value = torch.randn(
        (1, args.sequence_length, HIDDEN_SIZE), device=device, dtype=torch.bfloat16
    )
    scale = torch.randn((6, HIDDEN_SIZE), device=device, dtype=torch.bfloat16) * 0.05
    shift = torch.randn((6, HIDDEN_SIZE), device=device, dtype=torch.bfloat16) * 0.05
    indices = torch.arange(args.sequence_length, device=device, dtype=torch.int64) % 6

    with torch.no_grad():
        reference, candidate = module(value, scale, shift, indices)
    torch.cuda.synchronize(device)
    metrics = _metrics(reference, candidate)
    digest = hashlib.sha256(candidate.cpu().contiguous().view(torch.uint8).numpy()).hexdigest()
    local = {
        "rank": rank,
        "local_rank": local_rank,
        "live_weight": module.live_receipt,
        "metrics": metrics,
        "candidate_sha256": digest,
    }
    rows = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(rows, local)
    errors = []
    for row in rows:
        live = row["live_weight"]
        if live["weight_dtype"] != "torch.bfloat16":
            errors.append(f"rank{row['rank']} weight dtype {live['weight_dtype']}")
        if not live["weight_device"].startswith("cuda"):
            errors.append(f"rank{row['rank']} weight device {live['weight_device']}")
        if not live["weight_contiguous"] or live["weight_numel"] != HIDDEN_SIZE:
            errors.append(f"rank{row['rank']} invalid live weight {live}")
        if not row["metrics"]["finite"]:
            errors.append(f"rank{row['rank']} non-finite")
        if row["metrics"]["normalized_l2"] > 5e-5:
            errors.append(f"rank{row['rank']} nL2 {row['metrics']['normalized_l2']}")
        if row["metrics"]["cosine"] < 0.999999:
            errors.append(f"rank{row['rank']} cosine {row['metrics']['cosine']}")
        if row["metrics"]["max_abs"] > 0.0625:
            errors.append(f"rank{row['rank']} max_abs {row['metrics']['max_abs']}")
    if len({row["candidate_sha256"] for row in rows}) != 1:
        errors.append("candidate output SHA differs across ranks")

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

