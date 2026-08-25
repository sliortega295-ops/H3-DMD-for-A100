#!/usr/bin/env python3
"""Exercise the per-instance H3 forward patch on the pinned Diffusers class."""

from __future__ import annotations

import copy
import json

import torch
from torch.utils.checkpoint import checkpoint
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TransformerBlock

from h3_a100.fused_block_pointwise import install_fused_block_pointwise


class _Attention(torch.nn.Module):
    def forward(self, hidden_states, rotary_emb, attention_mask=None):
        del rotary_emb, attention_mask
        return hidden_states * torch.tensor(0.25, dtype=hidden_states.dtype, device=hidden_states.device)


class _FeedForward(torch.nn.Module):
    def forward(self, hidden_states):
        return hidden_states * torch.tensor(0.5, dtype=hidden_states.dtype, device=hidden_states.device)


class _AdaLN(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        generator = torch.Generator().manual_seed(20260825)
        table = torch.randn((6, 6, hidden_size), generator=generator, dtype=torch.bfloat16) * 0.05
        self.register_buffer("table", table)

    def forward(self, temb):
        del temb
        return tuple(self.table[index] for index in range(6))


class _Transformer(torch.nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList(blocks)


def make_block(hidden_size: int):
    block = MiniMaxH3TransformerBlock(
        hidden_size=hidden_size,
        num_attention_heads=2,
        attention_head_dim=64,
        ffn_dim=256,
        time_embed_dim=64,
        norm_eps=1e-5,
        qk_norm_eps=1e-5,
    )
    block.attn = _Attention()
    block.ff = _FeedForward()
    block.adaln_proj = _AdaLN(hidden_size)
    return block.to(device="cuda", dtype=torch.bfloat16)


def main():
    torch.manual_seed(20260825)
    hidden_size = 128
    sequence_length = 256
    reference = make_block(hidden_size)
    candidate = copy.deepcopy(reference)
    transformer = _Transformer([candidate] + [copy.deepcopy(candidate) for _ in range(49)])
    registration = install_fused_block_pointwise(
        transformer, enabled=True, grad_enabled=True
    )
    indices = torch.arange(sequence_length, device="cuda", dtype=torch.int64) % 6
    temb = torch.empty((6, 64), device="cuda", dtype=torch.bfloat16)
    hidden = torch.randn((1, sequence_length, hidden_size), device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        expected = reference(hidden, temb, indices, None)
        observed = transformer.transformer_blocks[0](hidden, temb, indices, None)
    if not torch.equal(expected, observed):
        raise RuntimeError("no-grad fused block output is not bitwise equal")

    reference_input = hidden.detach().clone().requires_grad_(True)
    candidate_input = hidden.detach().clone().requires_grad_(True)
    reference_output = reference(reference_input, temb, indices, None)
    candidate_output = transformer.transformer_blocks[0](candidate_input, temb, indices, None)
    upstream = torch.randn_like(reference_output)
    reference_output.backward(upstream)
    candidate_output.backward(upstream)
    if not torch.equal(reference_output, candidate_output):
        raise RuntimeError("grad-enabled patched block output changed")
    if not torch.equal(reference_input.grad, candidate_input.grad):
        raise RuntimeError("grad-enabled patched block gradient changed")

    # Validate entry accounting under the real non-reentrant checkpoint
    # mechanism.  Replay is allowed to early-stop before the Python block
    # wrapper returns, so one logical checkpointed block must count two entries
    # but only one set of pointwise backward calls.
    checkpoint_reference = hidden.detach().clone().requires_grad_(True)
    checkpoint_candidate = hidden.detach().clone().requires_grad_(True)
    before = registration.snapshot()
    expected_checkpoint = checkpoint(
        lambda value: reference(value, temb, indices, None),
        checkpoint_reference,
        use_reentrant=False,
    )
    observed_checkpoint = checkpoint(
        lambda value: transformer.transformer_blocks[0](value, temb, indices, None),
        checkpoint_candidate,
        use_reentrant=False,
    )
    checkpoint_upstream = torch.randn_like(expected_checkpoint)
    expected_checkpoint.backward(checkpoint_upstream)
    observed_checkpoint.backward(checkpoint_upstream)
    if not torch.equal(expected_checkpoint, observed_checkpoint):
        raise RuntimeError("checkpointed fused block output changed")
    if not torch.equal(checkpoint_reference.grad, checkpoint_candidate.grad):
        raise RuntimeError("checkpointed fused block gradient changed")
    after = registration.snapshot()
    checkpoint_delta = {key: int(after[key]) - int(before[key]) for key in after}
    expected_delta = {
        "fused_grad_block_calls": 2,
        "fused_grad_modulation_calls": 4,
        "fused_grad_residual_calls": 4,
        "fused_grad_modulation_backward_calls": 2,
        "fused_grad_residual_backward_calls": 2,
    }
    errors = [
        f"{key}={checkpoint_delta[key]} expected={value}"
        for key, value in expected_delta.items()
        if checkpoint_delta[key] != value
    ]
    if errors:
        raise RuntimeError("checkpoint pointwise accounting failed: " + "; ".join(errors))

    print(
        json.dumps(
            {
                "status": "PASS",
                "source_sha256": registration.source_sha256,
                "block_count": registration.block_count,
                "no_grad_output_bitwise_equal": True,
                "grad_output_bitwise_equal": True,
                "grad_input_bitwise_equal": True,
                "checkpoint_output_bitwise_equal": True,
                "checkpoint_grad_input_bitwise_equal": True,
                "checkpoint_delta": checkpoint_delta,
                "stats": registration.snapshot(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
