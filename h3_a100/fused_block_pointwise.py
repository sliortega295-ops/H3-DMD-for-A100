"""Opt-in exact no-grad pointwise fusion for pinned MiniMax-H3 blocks."""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import types
from typing import Any

import torch


PINNED_BLOCK_FORWARD_SHA256 = "ca6d9ca44871d0ed10fceaa4eecd184a7ac907cbd3e3b8fd9f761b91598a944d"
EXPECTED_BLOCK_COUNT = 50
EXPECTED_NOGRAD_BLOCK_CALLS_PER_CYCLE = 25 * EXPECTED_BLOCK_COUNT


@dataclasses.dataclass
class FusedPointwiseStats:
    fused_nograd_block_calls: int = 0
    reference_grad_block_calls: int = 0
    fused_modulation_calls: int = 0
    fused_residual_calls: int = 0


@dataclasses.dataclass
class FusedPointwiseRegistration:
    enabled: bool
    source_sha256: str | None
    block_count: int
    stats: FusedPointwiseStats

    def snapshot(self) -> dict[str, int]:
        return dataclasses.asdict(self.stats)

    def receipt(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "source_sha256": self.source_sha256,
            "block_count": self.block_count,
            "stats": self.snapshot(),
        }


def _source_sha256(function) -> str:
    return hashlib.sha256(inspect.getsource(function).encode()).hexdigest()


def _patched_forward(
    self,
    hidden_states: torch.Tensor,
    temb: torch.Tensor,
    adaln_indices: torch.Tensor,
    rotary_emb,
    attention_mask=None,
):
    stats: FusedPointwiseStats = self._h3_fused_pointwise_stats
    if torch.is_grad_enabled():
        stats.reference_grad_block_calls += 1
        return self._h3_original_forward(
            hidden_states,
            temb,
            adaln_indices,
            rotary_emb,
            attention_mask,
        )

    # Lazy import keeps the baseline/import-only path independent of Triton.
    from .triton_fused_pointwise import fused_modulate, fused_residual

    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(temb)

    residual = hidden_states
    norm_hidden_states = self.norm1(hidden_states)
    norm_hidden_states = fused_modulate(
        norm_hidden_states,
        scale_msa,
        shift_msa,
        adaln_indices,
    )
    stats.fused_modulation_calls += 1
    attn_output = self.attn(norm_hidden_states, rotary_emb, attention_mask)
    hidden_states = fused_residual(residual, gate_msa, attn_output, adaln_indices)
    stats.fused_residual_calls += 1

    residual = hidden_states
    norm_hidden_states = self.norm2(hidden_states)
    norm_hidden_states = fused_modulate(
        norm_hidden_states,
        scale_mlp,
        shift_mlp,
        adaln_indices,
    )
    stats.fused_modulation_calls += 1
    ff_output = self.ff(norm_hidden_states)
    hidden_states = fused_residual(residual, gate_mlp, ff_output, adaln_indices)
    stats.fused_residual_calls += 1
    stats.fused_nograd_block_calls += 1
    return hidden_states


def install_fused_block_pointwise(transformer: torch.nn.Module, *, enabled: bool) -> FusedPointwiseRegistration:
    """Patch the 50 pinned block instances without editing shared Diffusers."""

    blocks = list(getattr(transformer, "transformer_blocks", ()))
    if not enabled:
        return FusedPointwiseRegistration(False, None, len(blocks), FusedPointwiseStats())

    if len(blocks) != EXPECTED_BLOCK_COUNT:
        raise RuntimeError(
            f"H3 fused pointwise requires {EXPECTED_BLOCK_COUNT} blocks, got {len(blocks)}"
        )
    block_type = type(blocks[0])
    if any(type(block) is not block_type for block in blocks):
        raise RuntimeError("H3 fused pointwise requires one homogeneous pinned block type")
    source_sha256 = _source_sha256(block_type.forward)
    if source_sha256 != PINNED_BLOCK_FORWARD_SHA256:
        raise RuntimeError(
            "H3 block forward source identity mismatch: "
            f"observed={source_sha256} expected={PINNED_BLOCK_FORWARD_SHA256}"
        )

    stats = FusedPointwiseStats()
    for block in blocks:
        if hasattr(block, "_h3_original_forward"):
            raise RuntimeError("H3 fused pointwise was installed more than once")
        object.__setattr__(block, "_h3_original_forward", block.forward)
        object.__setattr__(block, "_h3_fused_pointwise_stats", stats)
        object.__setattr__(block, "forward", types.MethodType(_patched_forward, block))
    return FusedPointwiseRegistration(True, source_sha256, len(blocks), stats)


def validate_cycle(registration: FusedPointwiseRegistration, start: dict[str, int]) -> dict[str, int]:
    """Fail closed when any no-grad H3 block silently falls back to eager."""

    if not registration.enabled:
        return {}
    current = registration.snapshot()
    delta = {key: int(current[key]) - int(start.get(key, 0)) for key in current}
    expected = {
        "fused_nograd_block_calls": EXPECTED_NOGRAD_BLOCK_CALLS_PER_CYCLE,
        "fused_modulation_calls": 2 * EXPECTED_NOGRAD_BLOCK_CALLS_PER_CYCLE,
        "fused_residual_calls": 2 * EXPECTED_NOGRAD_BLOCK_CALLS_PER_CYCLE,
    }
    errors = [f"{key}={delta[key]} expected={value}" for key, value in expected.items() if delta[key] != value]
    if errors:
        raise RuntimeError(f"H3 fused pointwise cycle contract failed: {'; '.join(errors)}")
    return delta

