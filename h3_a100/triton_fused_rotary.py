"""Exact BF16 rotary fusion for MiniMax-H3 no-grad attention.

The pinned Diffusers implementation materializes a rotate-half tensor, two
BF16 products, their BF16 sum, and a final concatenation for every Q/K pair.
This kernel preserves those BF16 rounding boundaries while writing the final
``[rotary, pass-through]`` tensor directly.  It intentionally has no autograd
implementation; gradient-bearing forward and checkpoint replay remain on the
pinned upstream function.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _bf16_rotary_exact(value, rotated, cosine, sine):
    """Return ``bf16(bf16(value*cos) + bf16(rotated*sin))`` exactly."""

    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .b16 xh0,xh1,rh0,rh1,ch0,ch1,sh0,sh1,h0,h1,j0,j1,o0,o1;
          .reg .f32 xf0,xf1,rf0,rf1,cf0,cf1,sf0,sf1,p0,p1,q0,q1,t0,t1;
          mov.b32 {xh0,xh1}, $1;
          mov.b32 {rh0,rh1}, $2;
          mov.b32 {ch0,ch1}, $3;
          mov.b32 {sh0,sh1}, $4;
          cvt.f32.bf16 xf0, xh0; cvt.f32.bf16 xf1, xh1;
          cvt.f32.bf16 rf0, rh0; cvt.f32.bf16 rf1, rh1;
          cvt.f32.bf16 cf0, ch0; cvt.f32.bf16 cf1, ch1;
          cvt.f32.bf16 sf0, sh0; cvt.f32.bf16 sf1, sh1;
          mul.f32 p0, xf0, cf0; mul.f32 p1, xf1, cf1;
          cvt.rn.bf16.f32 h0, p0; cvt.rn.bf16.f32 h1, p1;
          mul.f32 q0, rf0, sf0; mul.f32 q1, rf1, sf1;
          cvt.rn.bf16.f32 j0, q0; cvt.rn.bf16.f32 j1, q1;
          cvt.f32.bf16 p0, h0; cvt.f32.bf16 p1, h1;
          cvt.f32.bf16 q0, j0; cvt.f32.bf16 q1, j1;
          add.f32 t0, p0, q0; add.f32 t1, p1, q1;
          cvt.rn.bf16.f32 o0, t0; cvt.rn.bf16.f32 o1, t1;
          mov.b32 $0, {o0,o1};
        }
        """,
        constraints="=r,r,r,r,r",
        args=[value, rotated, cosine, sine],
        dtype=tl.bfloat16,
        is_pure=True,
        pack=2,
    )


@triton.jit
def _rotary_kernel(
    hidden_states,
    cosine,
    sine,
    output,
    n_elements,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    valid = offsets < n_elements
    head_columns = offsets % head_dim
    rotary_mask = valid & (head_columns < rotary_dim)

    rows = offsets // head_dim
    # The frozen benchmark contract is B1/rank. Keeping the sequence extent
    # out of constexpr arguments prevents one Triton compilation per
    # rank-qualified packed sequence length.
    sequence_rows = rows // num_heads
    rotary_columns = head_columns
    half = rotary_dim // 2
    rotated_offsets = tl.where(
        rotary_columns < half,
        offsets + half,
        offsets - half,
    )

    values = tl.load(hidden_states + offsets, mask=valid, other=0.0)
    rotated = tl.load(hidden_states + rotated_offsets, mask=rotary_mask, other=0.0)
    rotated = tl.where(rotary_columns < half, -rotated, rotated).to(tl.bfloat16)
    table_offsets = sequence_rows * rotary_dim + rotary_columns
    cos_values = tl.load(cosine + table_offsets, mask=rotary_mask, other=0.0).to(tl.bfloat16)
    sin_values = tl.load(sine + table_offsets, mask=rotary_mask, other=0.0).to(tl.bfloat16)
    rotary_values = _bf16_rotary_exact(
        values.to(tl.bfloat16),
        rotated,
        cos_values,
        sin_values,
    )
    result = tl.where(rotary_mask, rotary_values, values)
    tl.store(output + offsets, result, mask=valid)


def fused_apply_rotary_emb(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply pinned MiniMax-H3 rotary math without intermediate tensors."""

    if not hidden_states.is_cuda or not cos.is_cuda or not sin.is_cuda:
        raise RuntimeError("H3 fused rotary requires CUDA tensors")
    if hidden_states.dtype != torch.bfloat16:
        raise RuntimeError(f"H3 fused rotary requires BF16 hidden states, got {hidden_states.dtype}")
    if hidden_states.ndim != 4 or cos.ndim != 2 or sin.ndim != 2:
        raise RuntimeError(
            "H3 fused rotary requires hidden=[B,S,NH,HD] and cos/sin=[S,R]"
        )
    if cos.shape != sin.shape or cos.device != hidden_states.device or sin.device != hidden_states.device:
        raise RuntimeError("H3 fused rotary cos/sin shape or device mismatch")
    batch, sequence_length, num_heads, head_dim = map(int, hidden_states.shape)
    if batch != 1:
        raise RuntimeError(f"H3 fused rotary is registered only for the frozen B1 contract, got B={batch}")
    rotary_dim = int(cos.shape[1])
    if int(cos.shape[0]) != sequence_length:
        raise RuntimeError(
            f"H3 fused rotary sequence mismatch hidden={sequence_length} cos={cos.shape[0]}"
        )
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
        raise RuntimeError(
            f"H3 fused rotary requires even 0 < rotary_dim <= head_dim, got {rotary_dim}/{head_dim}"
        )
    if cos.dtype not in {torch.float32, torch.bfloat16} or sin.dtype != cos.dtype:
        raise RuntimeError(f"H3 fused rotary requires matching FP32/BF16 cos/sin, got {cos.dtype}/{sin.dtype}")
    if not hidden_states.is_contiguous() or not cos.is_contiguous() or not sin.is_contiguous():
        raise RuntimeError("H3 fused rotary requires contiguous hidden/cos/sin tensors")
    n_elements = batch * sequence_length * num_heads * head_dim
    if n_elements % 2:
        raise RuntimeError(f"H3 fused rotary BF16 pack=2 requires even element count, got {n_elements}")
    major, minor = torch.cuda.get_device_capability(hidden_states.device)
    if (major, minor) < (8, 0):
        raise RuntimeError(f"H3 fused rotary requires SM80+, got sm_{major}{minor}")

    output = torch.empty_like(hidden_states)
    block_size = 1024
    _rotary_kernel[(triton.cdiv(n_elements, block_size),)](
        hidden_states,
        cos,
        sin,
        output,
        n_elements=n_elements,
        num_heads=num_heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        block_size=block_size,
        num_warps=8,
    )
    return output
