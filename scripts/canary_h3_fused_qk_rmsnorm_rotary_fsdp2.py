#!/usr/bin/env python3
"""Two-rank FSDP2 canary for live Q/K RMSNorm weight materialization."""

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

from h3_a100.triton_fused_rotary import (
    fused_apply_rotary_emb,
    fused_qk_rmsnorm_rotary,
)


HEADS = 56
HEAD_DIM = 128
ROTARY_DIM = 96
EPS = 1e-5


class _Unit(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.ones(HEAD_DIM, dtype=torch.bfloat16), requires_grad=False
        )
        self.live_receipt = None

    def forward(self, hidden, cos, sin):
        self.live_receipt = {
            "weight_dtype": str(self.weight.dtype),
            "weight_device": str(self.weight.device),
            "weight_type": type(self.weight).__name__,
            "weight_contiguous": bool(self.weight.is_contiguous()),
            "weight_numel": int(self.weight.numel()),
        }
        reference = fused_apply_rotary_emb(
            F.rms_norm(hidden, (HEAD_DIM,), weight=self.weight, eps=EPS), cos, sin
        )
        candidate = fused_qk_rmsnorm_rotary(
            hidden, self.weight, cos, sin, eps=EPS
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

    module = fully_shard(_Unit().to(device), reshard_after_forward=True)
    hidden = torch.randn(
        (1, args.sequence_length, HEADS, HEAD_DIM),
        device=device,
        dtype=torch.bfloat16,
    )
    cos = torch.randn((args.sequence_length, ROTARY_DIM), device=device, dtype=torch.float32)
    sin = torch.randn_like(cos)
    with torch.no_grad():
        reference, candidate = module(hidden, cos, sin)
    torch.cuda.synchronize(device)
    parity = _metrics(reference, candidate)
    digest = hashlib.sha256(candidate.cpu().contiguous().view(torch.uint8).numpy()).hexdigest()
    local = {
        "rank": rank,
        "local_rank": local_rank,
        "live_weight": module.live_receipt,
        "parity": parity,
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
        if not live["weight_contiguous"] or live["weight_numel"] != HEAD_DIM:
            errors.append(f"rank{row['rank']} invalid live weight {live}")
        if not row["parity"]["finite"]:
            errors.append(f"rank{row['rank']} non-finite")
        if row["parity"]["normalized_l2"] > 5e-5:
            errors.append(f"rank{row['rank']} nL2 {row['parity']['normalized_l2']}")
        if row["parity"]["cosine"] < 0.999999:
            errors.append(f"rank{row['rank']} cosine {row['parity']['cosine']}")
        if row["parity"]["max_abs"] > 0.0625:
            errors.append(f"rank{row['rank']} max_abs {row['parity']['max_abs']}")
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
