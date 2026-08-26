# Iteration 393: FA3 split2 for grad/checkpoint replay

## Result

The independent grad/replay extension passed its real-kernel canary and the exact world16 workload. It is a **small provisional improvement**, not a variance-qualified speed claim.

| Run | FA3 no-grad | FA3 grad/replay | One-cycle wall | Peak allocated | Status |
|---|---:|---:|---:|---:|---|
| Iteration 390 parent | split2 | split1 | 813.609 s | 33.39 GiB | PASS |
| Iteration 393 | split2 | split2 | 812.029 s | 33.39 GiB | PASS |

Delta: **-1.580 s (-0.194%)**, or 1.00195x in this single matched run.

## Canary

On one owned A100 at the production attention shape `[1,37760,56,128]`, direct forward+backward measured 747.688 ms for split1 and 689.451 ms for split2. Split1 vs split2 output had max-abs 2.44e-4, nL2 0.0022124, cosine 0.99999757. Q/K/V gradient nL2 was at most 0.0009843 and cosine at least 0.99999952. The wrapper reproduced direct split2 output bitwise; its non-reentrant checkpoint invocation rewrote both original forward and replay (2 calls, 0 routing errors).

## Workload gates

- 16/16 ranks; 96 rank-qualified sample identities.
- Application census: 24/6/1 forwards, 1/5/0 grad-forwards, 1/5 backwards.
- Grid replay: 300 wraps, 600 scopes, 600 hits, 0 misses.
- FA3: 1,300 no-grad + 624 grad/replay calls; all 1,924 rewritten to split2; 0 unexpected inputs.
- Every other current fusion/LoRA counter passed unchanged.
- No CUDA/cgroup OOM; postflight found 16 idle GPUs and all locks free.

## Decision boundary

Keep the switch as the current best provisional path because it did not regress the matched run. The measured gain is only 1.58 s and may be ordinary run variance, so it must not be described as a formal speedup. The evidence does not justify a custom FlashAttention kernel rewrite.
