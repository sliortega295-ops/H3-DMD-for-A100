# MiniMax-H3 grad/replay Q/K RMSNorm + rotary fusion (2026-08-26)

## Scope

This bounded candidate extends the validated no-grad Q/K fusion to the
grad-enabled and non-reentrant checkpoint-replay path. It replaces the frozen
affine RMSNorm input-gradient followed by exact BF16 rotary backward with one
Triton kernel per Q or K. The feature is default-off and requires both existing
Q/K fusion and exact grad rotary.

No schedule, model, trace, sample, Grid-1000 key, attention, loss, update,
FSDP2, activation, or RNG contract changed.

## Canary gates

- Production tensor `[1,37760,56,128]`, BF16: PASS.
- Native reference forward+backward median: 22.1819 ms.
- Candidate median: 4.1615 ms.
- Output/gradient normalized L2: 1.0092e-5 / 1.0557e-5.
- Two-rank FSDP2 live frozen-weight canary: PASS; output and gradient SHA were
  identical across ranks.
- Focused tests passed on both cluster nodes.

The first production canary was preserved as a diagnostic failure: its
reference accidentally invoked a raw Triton forward that intentionally has no
autograd edge. Commit `b5c8aa3` changes only the canary reference to the pinned
Diffusers implementation; it does not change training math or the candidate
kernel.

## World16 result

| Run | Head | Time (s) | Status |
|---|---|---:|---|
| Iteration 383 parent | `0f596ca` | 823.037 | PASS, one run |
| Iteration 386 candidate | `b5c8aa3` | **820.008** | PASS, one run |

Provisional delta: **-3.029 s (-0.368%)**, or **1.003694x**.

Iteration 386 passed:

- 16/16 matched distributed workers;
- application forwards 24/6/1, grad forwards 1/5/0, backwards 1/5;
- 96 rank-qualified sample identities;
- Grid replay 300 wraps, 600 scopes, 600 hits, 0 misses;
- per rank: 600 fused grad attention calls, 1,200 fused Q/K forward calls,
  600 fused Q/K backward calls, and zero reference grad attention calls;
- parent exact rotary census remained 2,500 no-grad, 1,200 grad/replay forward,
  600 backward, and zero reference-grad calls;
- no CUDA/cgroup OOM; rank-0 peak allocated/reserved 34.77/36.65 GiB;
- all 16 GPUs idle and all three atomic locks free after teardown.

This is one unprofiled run. It promotes the bounded mechanism only; it is not a
median or variance-qualified speedup.

Persistent raw evidence:

- `iteration_385_qk_rmsnorm_rotary_grad_canary_20260826`
- `iteration_386_qk_rmsnorm_rotary_grad_world16_20260826`
