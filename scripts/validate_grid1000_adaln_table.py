#!/usr/bin/env python3
"""Validate cached Grid-1000 modulation against original H3 AdaLN weights."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from diffusers.models.modeling_utils import get_parameter_dtype

from h3_a100.grid_adaln import GridAdaLNController
from lightx2v_train.model_zoo.native.minimax_h3 import load_minimax_h3_transformer


def _shift_sigma(sigma: torch.Tensor, shift: float) -> torch.Tensor:
    return shift * sigma / (1.0 + (shift - 1.0) * sigma)


def _pair(base: torch.Tensor, video_shift: float, audio_shift: float) -> torch.Tensor:
    base = base.to(torch.float32).cpu()
    vt = torch.tensor(1.0, dtype=torch.float32) - _shift_sigma(base, video_shift)
    at = torch.tensor(1.0, dtype=torch.float32) - _shift_sigma(base, audio_shift)
    return torch.sort(torch.stack((vt, at))).values


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--random-grid-entries", type=int, default=8)
    parser.add_argument("--blocks-per-entry", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--attention-backend", default="_flash_3_hub")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    controller = GridAdaLNController(args.manifest, pin_memory=False)
    transformer = load_minimax_h3_transformer(
        str(args.model_path),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        attention_backend=args.attention_backend,
    )
    device = torch.device(args.device)
    transformer.time_proj.to(device)
    transformer.time_embedder.to(device)
    blocks = list(transformer.transformer_blocks)
    rng = random.Random(args.seed)

    rollout = [torch.tensor(v, dtype=torch.float32) for v in payload["rollout_base_sigmas"]]
    grid = [torch.tensor(v, dtype=torch.float32) for v in payload["grid_base_sigmas"]]
    grid_ids = rng.sample(range(len(grid)), min(args.random_grid_entries, len(grid)))
    selections = [(index, rollout[index]) for index in range(len(rollout))]
    selections += [(len(rollout) + index, grid[index]) for index in grid_ids]

    failures = []
    checked = 0
    for entry_index, base in selections:
        timesteps = _pair(base, float(payload["video_shift"]), float(payload["audio_shift"]))
        with torch.no_grad():
            temb = transformer.time_proj(timesteps.to(device))
            temb = transformer.time_embedder(
                temb.to(get_parameter_dtype(transformer.time_embedder))
            )
        block_ids = rng.sample(range(50), min(args.blocks_per_entry, 50))
        for block_index in block_ids:
            projection = blocks[block_index].adaln_proj.to(device)
            with torch.no_grad():
                chunks = projection(temb)
                expected = torch.stack(
                    [
                        # step0 has one runtime timestep but the fixed-shape
                        # table stores it twice; duplicate here to compare the
                        # same six-row representation.
                        (
                            chunk.repeat(2, 1)
                            if chunk.shape[0] == 3
                            else chunk
                        )
                        for chunk in chunks
                    ],
                    dim=0,
                ).to(torch.bfloat16).cpu()
            actual = controller._table[entry_index, block_index].clone()
            projection.to("cpu")
            checked += 1
            if expected.shape != actual.shape or not torch.equal(expected, actual):
                max_abs = None
                if expected.shape == actual.shape:
                    max_abs = float((expected.float() - actual.float()).abs().max().item())
                failures.append(
                    {
                        "entry": entry_index,
                        "block": block_index,
                        "expected_shape": list(expected.shape),
                        "actual_shape": list(actual.shape),
                        "max_abs": max_abs,
                    }
                )
            torch.cuda.empty_cache()

    report = {
        "checked": checked,
        "failures": failures,
        "status": "PASS" if not failures else "FAIL",
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
