#!/usr/bin/env python3
"""Build the full MiniMax-H3 Grid-1000 AdaLN lookup table on one GPU.

The builder is intentionally offline/cold-start work.  It loads the H3
checkpoint on CPU, keeps the small timestep MLP on one GPU, then processes one
~520 MB AdaLN projection at a time.  The output is a single ~19 GiB BF16 mmap
plus a small JSON manifest.  No model weights are written to the repository.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import torch
from diffusers.models.modeling_utils import get_parameter_dtype
from loguru import logger

from h3_a100.grid_adaln import SCHEMA_VERSION
from lightx2v_train.model_zoo.native.minimax_h3 import load_minimax_h3_transformer


def _shift_sigma(sigma: torch.Tensor, shift: float) -> torch.Tensor:
    return shift * sigma / (1.0 + (shift - 1.0) * sigma)


def _f32_bits(values: torch.Tensor) -> list[int]:
    return [
        int(value)
        for value in values.detach().to(dtype=torch.float32, device="cpu").contiguous().view(torch.int32).tolist()
    ]


def _timestep_pair(base_sigma: torch.Tensor, video_shift: float, audio_shift: float) -> torch.Tensor:
    base_sigma = base_sigma.to(device="cpu", dtype=torch.float32)
    video_sigma = _shift_sigma(base_sigma, video_shift)
    audio_sigma = _shift_sigma(base_sigma, audio_shift)
    # Match build_row_timesteps: subtraction is explicitly float32 and
    # torch.unique(..., sorted=True) orders the two AV timesteps.
    video_t = (torch.tensor(1.0, dtype=torch.float32) - video_sigma).to(torch.float32)
    audio_t = (torch.tensor(1.0, dtype=torch.float32) - audio_sigma).to(torch.float32)
    return torch.sort(torch.stack((video_t, audio_t))).values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grid-size", type=int, default=1000)
    parser.add_argument("--renoise-sigma-min", type=float, default=0.02)
    parser.add_argument("--renoise-sigma-max", type=float, default=0.98)
    parser.add_argument("--video-shift", type=float, default=6.0)
    parser.add_argument("--audio-shift", type=float, default=3.0)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attention-backend", default="_flash_3_hub")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.grid_size != 1000:
        raise ValueError("This experiment builder is intentionally fixed to Grid-1000")
    if not 0.0 <= args.renoise_sigma_min < args.renoise_sigma_max <= 1.0:
        raise ValueError("invalid renoise sigma interval")
    if args.num_inference_steps != 4:
        raise ValueError("matched H3 experiment requires four rollout evaluations")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Grid-1000 builder requires one CUDA GPU")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    binary_path = output_dir / "adaln_grid1000.bf16.bin"
    manifest_path = output_dir / "adaln_grid1000.json"
    if not args.overwrite and (binary_path.exists() or manifest_path.exists()):
        raise FileExistsError("Grid-1000 output already exists; use --overwrite explicitly")
    if args.overwrite:
        binary_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)

    logger.info("loading MiniMax-H3 on CPU from {}", args.model_path)
    transformer = load_minimax_h3_transformer(
        str(args.model_path),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        attention_backend=args.attention_backend,
    )
    blocks = list(transformer.transformer_blocks)
    if len(blocks) != 50:
        raise RuntimeError(f"expected 50 H3 blocks, got {len(blocks)}")
    hidden_size = int(getattr(blocks[0].adaln_proj, "hidden_size", 5376))
    dropped_parameter_numel = sum(
        parameter.numel()
        for block in blocks
        for parameter in block.adaln_proj.parameters()
    )

    rollout_base = torch.linspace(
        1.0, 0.0, args.num_inference_steps + 1, dtype=torch.float32
    )[:-1]
    grid_base = torch.linspace(
        args.renoise_sigma_min,
        args.renoise_sigma_max,
        args.grid_size,
        dtype=torch.float32,
    )
    base_sigmas = torch.cat((rollout_base, grid_base), dim=0)
    timestep_pairs = torch.stack(
        [
            _timestep_pair(base, args.video_shift, args.audio_shift)
            for base in base_sigmas
        ],
        dim=0,
    )
    num_entries = int(timestep_pairs.shape[0])
    if num_entries != args.num_inference_steps + args.grid_size:
        raise AssertionError("entry count mismatch")
    timestep_bits = [_f32_bits(row) for row in timestep_pairs]
    if len({tuple(row) for row in timestep_bits}) != num_entries:
        raise RuntimeError("rollout/grid timestep pairs unexpectedly collide")

    # Precompute the shared timestep embedding for all 2*1004 AV timesteps.
    transformer.time_proj.to(device)
    transformer.time_embedder.to(device)
    with torch.no_grad():
        flat_timesteps = timestep_pairs.reshape(-1).to(device)
        temb = transformer.time_proj(flat_timesteps)
        temb = transformer.time_embedder(
            temb.to(get_parameter_dtype(transformer.time_embedder))
        )
    if temb.shape[0] != num_entries * 2:
        raise RuntimeError(f"unexpected timestep embedding shape {tuple(temb.shape)}")

    shape = (num_entries, 50, 6, 6, hidden_size)
    numel = 1
    for value in shape:
        numel *= int(value)
    table = torch.from_file(
        str(binary_path), shared=True, size=numel, dtype=torch.bfloat16
    ).reshape(shape)
    logger.info(
        "building table shape={} bytes={:.2f} GiB dropped_adaln_params={:.3f}B",
        shape,
        binary_path.stat().st_size / 1024**3,
        dropped_parameter_numel / 1e9,
    )

    with torch.no_grad():
        for block_index, block in enumerate(blocks):
            projection = block.adaln_proj.to(device)
            chunks = projection(temb)
            if len(chunks) != 6:
                raise RuntimeError(f"block {block_index} returned {len(chunks)} AdaLN chunks")
            # projection(all 2-entry timesteps) returns six tensors of
            # [num_entries*6, hidden].  Regroup into
            # [entry, modulation_chunk, 2_timesteps*3_modalities, hidden].
            modulation = torch.stack(
                [chunk.reshape(num_entries, 6, hidden_size) for chunk in chunks],
                dim=1,
            ).to(dtype=torch.bfloat16)
            table[:, block_index].copy_(modulation.to(device="cpu"))
            projection.to("cpu")
            del chunks, modulation
            torch.cuda.empty_cache()
            logger.info("Grid-1000 AdaLN block {}/50 written", block_index + 1)

    # Drop mappings before fsync so dirty pages can be flushed.
    del table
    del temb
    gc.collect()
    torch.cuda.empty_cache()
    try:
        os.sync()
    except AttributeError:
        pass

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "binary_file": binary_path.name,
        "grid_size": int(args.grid_size),
        "num_rollout_entries": int(args.num_inference_steps),
        "num_entries": num_entries,
        "num_blocks": 50,
        "hidden_size": hidden_size,
        "rows_per_entry": 6,
        "modulation_chunks": 6,
        "dtype": "torch.bfloat16",
        "renoise_sigma_min": float(args.renoise_sigma_min),
        "renoise_sigma_max": float(args.renoise_sigma_max),
        "video_shift": float(args.video_shift),
        "audio_shift": float(args.audio_shift),
        "num_inference_steps": int(args.num_inference_steps),
        "rollout_base_sigmas": [float(value) for value in rollout_base.tolist()],
        "grid_base_sigmas": [float(value) for value in grid_base.tolist()],
        "timestep_bits": timestep_bits,
        "dropped_adaln_parameter_numel": int(dropped_parameter_numel),
        "binary_bytes": int(binary_path.stat().st_size),
        "model_path": str(args.model_path.resolve()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("Grid-1000 manifest written: {}", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
