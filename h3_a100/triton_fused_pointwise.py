"""Bitwise BF16 pointwise kernels for MiniMax-H3 no-grad blocks.

The upstream block materializes full-sequence AdaLN tables with ``index_select``
and then launches separate BF16 pointwise kernels for every multiply and add.
For the fixed A100 workload those tensors are hundreds of MiB.  These kernels
perform the same BF16 rounding sequence while gathering the six-row AdaLN table
directly, avoiding the temporary full-sequence tensors.

This module is imported lazily only when the opt-in production flag is enabled;
the baseline path therefore does not acquire a Triton dependency at import time.
The kernels intentionally have no autograd implementation.  Gradient-bearing
forward and checkpoint replay keep using the pinned upstream implementation.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _bf16_modulate_exact(value, scale, shift):
    """Return ``value * (1 + scale) + shift`` with eager BF16 rounding.

    A100 does not expose scalar BF16 add/mul instructions.  Eager PyTorch
    converts each BF16 operand to FP32, rounds the result back to BF16 at every
    kernel boundary, then converts it to FP32 for the next operation.  The
    explicit conversion chain below preserves that order inside one kernel.
    Two BF16 lanes are packed into one 32-bit inline-assembly invocation.
    """

    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .b16 vh0,vh1,sh0,sh1,bh0,bh1,h0,h1,o0,o1;
          .reg .f32 vf0,vf1,sf0,sf1,bf0,bf1,t0,t1,p0,p1;
          mov.b32 {vh0,vh1}, $1;
          mov.b32 {sh0,sh1}, $2;
          mov.b32 {bh0,bh1}, $3;
          cvt.f32.bf16 vf0, vh0; cvt.f32.bf16 vf1, vh1;
          cvt.f32.bf16 sf0, sh0; cvt.f32.bf16 sf1, sh1;
          cvt.f32.bf16 bf0, bh0; cvt.f32.bf16 bf1, bh1;
          add.f32 t0, sf0, 1.0; add.f32 t1, sf1, 1.0;
          cvt.rn.bf16.f32 h0, t0; cvt.rn.bf16.f32 h1, t1;
          cvt.f32.bf16 t0, h0; cvt.f32.bf16 t1, h1;
          mul.f32 p0, vf0, t0; mul.f32 p1, vf1, t1;
          cvt.rn.bf16.f32 h0, p0; cvt.rn.bf16.f32 h1, p1;
          cvt.f32.bf16 p0, h0; cvt.f32.bf16 p1, h1;
          add.f32 t0, p0, bf0; add.f32 t1, p1, bf1;
          cvt.rn.bf16.f32 o0, t0; cvt.rn.bf16.f32 o1, t1;
          mov.b32 $0, {o0,o1};
        }
        """,
        constraints="=r,r,r,r",
        args=[value, scale, shift],
        dtype=tl.bfloat16,
        is_pure=True,
        pack=2,
    )


@triton.jit
def _bf16_residual_exact(residual, gate, branch):
    """Return ``residual + gate * branch`` with eager BF16 rounding."""

    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .b16 rh0,rh1,gh0,gh1,bh0,bh1,h0,h1,o0,o1;
          .reg .f32 rf0,rf1,gf0,gf1,bf0,bf1,p0,p1,t0,t1;
          mov.b32 {rh0,rh1}, $1;
          mov.b32 {gh0,gh1}, $2;
          mov.b32 {bh0,bh1}, $3;
          cvt.f32.bf16 rf0, rh0; cvt.f32.bf16 rf1, rh1;
          cvt.f32.bf16 gf0, gh0; cvt.f32.bf16 gf1, gh1;
          cvt.f32.bf16 bf0, bh0; cvt.f32.bf16 bf1, bh1;
          mul.f32 p0, gf0, bf0; mul.f32 p1, gf1, bf1;
          cvt.rn.bf16.f32 h0, p0; cvt.rn.bf16.f32 h1, p1;
          cvt.f32.bf16 p0, h0; cvt.f32.bf16 p1, h1;
          add.f32 t0, rf0, p0; add.f32 t1, rf1, p1;
          cvt.rn.bf16.f32 o0, t0; cvt.rn.bf16.f32 o1, t1;
          mov.b32 $0, {o0,o1};
        }
        """,
        constraints="=r,r,r,r",
        args=[residual, gate, branch],
        dtype=tl.bfloat16,
        is_pure=True,
        pack=2,
    )


@triton.jit
def _modulate_kernel(
    value,
    scale,
    shift,
    indices,
    output,
    n_elements: tl.constexpr,
    hidden_size: tl.constexpr,
    sequence_length: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < n_elements
    rows = offsets // hidden_size
    columns = offsets - rows * hidden_size
    sequence_rows = rows % sequence_length
    table_rows = tl.load(indices + sequence_rows, mask=mask, other=0).to(tl.int64)
    table_offsets = table_rows * hidden_size + columns
    values = tl.load(value + offsets, mask=mask, other=0.0)
    scales = tl.load(scale + table_offsets, mask=mask, other=0.0)
    shifts = tl.load(shift + table_offsets, mask=mask, other=0.0)
    result = _bf16_modulate_exact(values, scales, shifts)
    tl.store(output + offsets, result, mask=mask)


@triton.jit
def _residual_kernel(
    residual,
    gate,
    branch,
    indices,
    output,
    n_elements: tl.constexpr,
    hidden_size: tl.constexpr,
    sequence_length: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < n_elements
    rows = offsets // hidden_size
    columns = offsets - rows * hidden_size
    sequence_rows = rows % sequence_length
    table_rows = tl.load(indices + sequence_rows, mask=mask, other=0).to(tl.int64)
    table_offsets = table_rows * hidden_size + columns
    residual_values = tl.load(residual + offsets, mask=mask, other=0.0)
    gates = tl.load(gate + table_offsets, mask=mask, other=0.0)
    branch_values = tl.load(branch + offsets, mask=mask, other=0.0)
    result = _bf16_residual_exact(residual_values, gates, branch_values)
    tl.store(output + offsets, result, mask=mask)


def _validate_common(
    value: torch.Tensor,
    table: torch.Tensor,
    indices: torch.Tensor,
) -> tuple[int, int, int]:
    tensors = (value, table, indices)
    if any(not tensor.is_cuda for tensor in tensors):
        raise RuntimeError("H3 fused pointwise requires CUDA tensors")
    if value.dtype != torch.bfloat16 or table.dtype != value.dtype:
        raise RuntimeError("H3 fused pointwise requires BF16 activation and table")
    if indices.dtype not in {torch.int32, torch.int64}:
        raise RuntimeError(f"H3 fused pointwise requires int32/int64 indices, got {indices.dtype}")
    if value.ndim != 3 or table.ndim != 2 or indices.ndim != 1:
        raise RuntimeError("H3 fused pointwise requires value=[B,S,H], table=[K,H], indices=[S]")
    batch, sequence_length, hidden_size = value.shape
    if sequence_length != indices.numel():
        raise RuntimeError(
            f"H3 fused pointwise sequence mismatch value={sequence_length} indices={indices.numel()}"
        )
    if table.shape[1] != hidden_size:
        raise RuntimeError(
            f"H3 fused pointwise table mismatch table={tuple(table.shape)} hidden={hidden_size}"
        )
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise RuntimeError("H3 fused pointwise requires contiguous tensors")
    n_elements = int(batch * sequence_length * hidden_size)
    if n_elements % 2 != 0:
        raise RuntimeError(f"H3 fused BF16 pack=2 requires an even element count, got {n_elements}")
    major, minor = torch.cuda.get_device_capability(value.device)
    if (major, minor) < (8, 0):
        raise RuntimeError(f"H3 fused pointwise requires SM80+, got sm_{major}{minor}")
    return n_elements, int(sequence_length), int(hidden_size)


def fused_modulate(
    value: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    n_elements, sequence_length, hidden_size = _validate_common(value, scale, indices)
    if shift.shape != scale.shape or shift.dtype != value.dtype or not shift.is_cuda or not shift.is_contiguous():
        raise RuntimeError(
            f"H3 fused modulation shift mismatch scale={tuple(scale.shape)} shift={tuple(shift.shape)}"
        )
    output = torch.empty_like(value)
    block_size = 256
    _modulate_kernel[(triton.cdiv(n_elements, block_size),)](
        value,
        scale,
        shift,
        indices,
        output,
        n_elements=n_elements,
        hidden_size=hidden_size,
        sequence_length=sequence_length,
        block_size=block_size,
    )
    return output


def fused_residual(
    residual: torch.Tensor,
    gate: torch.Tensor,
    branch: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    n_elements, sequence_length, hidden_size = _validate_common(residual, gate, indices)
    if branch.shape != residual.shape or branch.dtype != residual.dtype or not branch.is_cuda or not branch.is_contiguous():
        raise RuntimeError(
            f"H3 fused residual branch mismatch residual={tuple(residual.shape)} branch={tuple(branch.shape)}"
        )
    output = torch.empty_like(residual)
    block_size = 256
    _residual_kernel[(triton.cdiv(n_elements, block_size),)](
        residual,
        gate,
        branch,
        indices,
        output,
        n_elements=n_elements,
        hidden_size=hidden_size,
        sequence_length=sequence_length,
        block_size=block_size,
    )
    return output
