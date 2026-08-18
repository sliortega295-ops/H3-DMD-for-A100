#!/usr/bin/env python3
"""Fail-fast environment/source checks before a controlled H3 DMD run."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

EXPECTED_DIFFUSERS_VERSION = "0.40.0.dev0"
EXPECTED_DIFFUSERS_SOURCE_REVISION = "9284607295a09f759aadd65ed08f48b35feea6d9"
EXPECTED_ATTENTION_BACKEND = "_flash_3_hub"


def _gib(value: int) -> float:
    return value / 1024**3


def _host_memory_gib() -> float | None:
    path = Path("/proc/meminfo")
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) / 1024**2
    return None


def _git_head(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _module_source_revision(module) -> str | None:
    module_path = Path(getattr(module, "__file__", "")).resolve()
    for parent in (module_path.parent, *module_path.parents):
        if (parent / ".git").exists():
            return _git_head(parent)
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--lightx2v-root", type=Path, default=None)
    parser.add_argument("--expected-gpus", type=int, default=8)
    parser.add_argument("--min-host-memory-gib", type=float, default=620.0)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    lightx2v_root = (
        args.lightx2v_root
        or Path(os.environ.get("LIGHTX2V_ROOT", repo_root / "third_party" / "LightX2V"))
    ).resolve()
    expected_commit = (repo_root / "UPSTREAM_COMMIT").read_text(encoding="utf-8").strip()

    errors: list[str] = []
    warnings: list[str] = []
    report: dict[str, object] = {
        "repo_root": str(repo_root),
        "lightx2v_root": str(lightx2v_root),
        "expected_upstream_commit": expected_commit,
        "expected_diffusers_version": EXPECTED_DIFFUSERS_VERSION,
        "expected_diffusers_source_revision": EXPECTED_DIFFUSERS_SOURCE_REVISION,
        "expected_attention_backend": EXPECTED_ATTENTION_BACKEND,
    }

    if not lightx2v_root.is_dir():
        errors.append(f"LightX2V checkout not found: {lightx2v_root}")
    else:
        head = _git_head(lightx2v_root)
        report["lightx2v_head"] = head
        if head != expected_commit:
            errors.append(f"LightX2V HEAD is {head}; expected pinned commit {expected_commit}")

    actual_attention_backend = os.environ.get("H3_ATTN_BACKEND", EXPECTED_ATTENTION_BACKEND)
    report["attention_backend"] = actual_attention_backend
    if actual_attention_backend != EXPECTED_ATTENTION_BACKEND:
        errors.append(
            f"Controlled timing requires H3_ATTN_BACKEND={EXPECTED_ATTENTION_BACKEND}, "
            f"got {actual_attention_backend}"
        )

    try:
        import torch

        report["torch"] = torch.__version__
        report["cuda_available"] = torch.cuda.is_available()
        if not torch.cuda.is_available():
            errors.append("CUDA is not available")
        else:
            count = torch.cuda.device_count()
            report["gpu_count"] = count
            report["gpus"] = [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": torch.cuda.get_device_capability(index),
                    "memory_gib": round(_gib(torch.cuda.get_device_properties(index).total_memory), 2),
                }
                for index in range(count)
            ]
            if count != args.expected_gpus:
                errors.append(f"Visible GPU count is {count}; expected {args.expected_gpus}")
            for gpu in report["gpus"]:
                if "A100" not in gpu["name"] and "A800" not in gpu["name"]:
                    warnings.append(f"Unexpected GPU for the validated path: {gpu['name']}")
        from torch.distributed.fsdp import fully_shard  # noqa: F401
    except Exception as exc:  # pragma: no cover - cluster-only diagnostics
        errors.append(f"PyTorch/FSDP2 check failed: {exc}")

    for package in ("diffusers", "peft", "safetensors", "omegaconf"):
        try:
            report[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"Required package is missing: {package}")

    try:
        diffusers = importlib.import_module("diffusers")
        actual_version = str(getattr(diffusers, "__version__", ""))
        actual_revision = _module_source_revision(diffusers)
        report["diffusers_version"] = actual_version
        report["diffusers_source_revision"] = actual_revision
        if actual_version != EXPECTED_DIFFUSERS_VERSION:
            errors.append(
                f"Diffusers version is {actual_version}; expected {EXPECTED_DIFFUSERS_VERSION}"
            )
        if actual_revision != EXPECTED_DIFFUSERS_SOURCE_REVISION:
            errors.append(
                "Diffusers source revision is "
                f"{actual_revision}; expected {EXPECTED_DIFFUSERS_SOURCE_REVISION}"
            )
        cls = getattr(diffusers, "MiniMaxH3Transformer3DModel")
        for method in ("add_adapter", "set_adapter", "disable_adapters", "set_attention_backend"):
            if not hasattr(cls, method):
                errors.append(f"Diffusers MiniMaxH3Transformer3DModel lacks {method}()")
    except Exception as exc:
        errors.append(f"MiniMax-H3 Diffusers API/source check failed: {exc}")

    model_root = Path(os.environ.get("MINIMAX_H3_MODEL_PATH", "/models/MiniMax-H3"))
    transformer_config = model_root / "transformer" / "config.json"
    report["model_root"] = str(model_root)
    if not transformer_config.is_file():
        errors.append(f"Converted H3 transformer config not found: {transformer_config}")
    else:
        config = json.loads(transformer_config.read_text(encoding="utf-8"))
        if config.get("_class_name") != "MiniMaxH3Transformer3DModel":
            errors.append(
                "MINIMAX_H3_MODEL_PATH points to the original layout; LightX2V training needs the "
                "converted Diffusers transformer layout"
            )

    prompt_cache = Path(os.environ.get("H3_PROMPT_CACHE", "/datasets/minimax_h3_prompt_cache"))
    report["prompt_cache"] = str(prompt_cache)
    if not prompt_cache.exists():
        errors.append(f"Prompt cache not found: {prompt_cache}")

    host_memory = _host_memory_gib()
    report["host_memory_gib"] = host_memory
    if host_memory is not None and host_memory < args.min_host_memory_gib:
        warnings.append(
            f"Host RAM is {host_memory:.1f} GiB; current CPU-first loader may have a high cold-start peak"
        )

    report["warnings"] = warnings
    report["errors"] = errors
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
