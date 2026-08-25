#!/usr/bin/env python3
"""Exercise the per-instance H3 forward patch on the pinned Diffusers class."""

from __future__ import annotations

import copy
import json

import torch
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
    registration = install_fused_block_pointwise(transformer, enabled=True)
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
    reference_output.float().sum().backward()
    candidate_output.float().sum().backward()
    if not torch.equal(reference_output, candidate_output):
        raise RuntimeError("grad-enabled patched block output changed")
    if not torch.equal(reference_input.grad, candidate_input.grad):
        raise RuntimeError("grad-enabled patched block gradient changed")

    print(
        json.dumps(
            {
                "status": "PASS",
                "source_sha256": registration.source_sha256,
                "block_count": registration.block_count,
                "no_grad_output_bitwise_equal": True,
                "grad_output_bitwise_equal": True,
                "grad_input_bitwise_equal": True,
                "stats": registration.snapshot(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
