# MiniMax-H3 no-grad Q/K RMSNorm + rotary fusion (2026-08-26)

## Scope

This bounded candidate starts from the validated Iteration 379 world16 workload and changes only the physical implementation of frozen, no-grad main-block attention:

```text
Q/K projection -> RMSNorm -> exact BF16 rotary
```

becomes one Triton kernel per Q or K. Token-refiner attention, all grad-enabled forwards, non-reentrant checkpoint replay, attention, projections, losses, update order, samples, Grid-1000 keys, and FSDP2 placement remain unchanged. The feature is default-off and requires the existing exact rotary feature.

## Gates

- Production tensor: `[1,37760,56,128]`, BF16, rotary dimension 96.
- Parent (PyTorch RMSNorm + validated fused rotary): 3.496960 ms median.
- Candidate: 2.075648 ms median; paired median delta -1.418240 ms/call.
- Numerical: max absolute 0.03125, normalized L2 1.0092e-5, cosine 0.999999999949.
- Two-rank FSDP2: live RMSNorm weights on both ranks were full, contiguous CUDA BF16 tensors with identical output hashes.
- Focused tests passed on both nodes.

## Preserved failed run

Iteration 382 completed the matched workload, but the pre-existing exact-rotary receipt observed zero no-grad calls because the new fused kernel subsumed those operations without forwarding its physical-call census. It failed closed after compute. Its 823.133 s diagnostic interval is not a performance result.

The bounded repair forwards two physical exact-rotary calls into the installed parent registration per fused attention call. It does not alter tensor math or launch another kernel. A unit test exercises the nested census.

## World16 result

| Run | Head | Time (s) | Status |
|---|---|---:|---|
| Iteration 379 parent | `7a9e0e3` | 825.947 | PASS, one run |
| Iteration 383 candidate | `0f596ca` | **823.037** | PASS, one run |

Provisional delta: **-2.910 s (-0.352%)**, or **1.003536x**.

Iteration 383 passed:

- 16/16 distributed workers (node launcher exits 0/0);
- application forwards 24/6/1, grad forwards 1/5/0, backwards 1/5;
- 96 rank-qualified sample identities;
- Grid replay 300 wraps, 600 scopes, 600 hits, 0 misses;
- per rank: 1,250 no-grad attention calls, 2,500 fused Q/K operations, 600 reference grad/replay attention calls;
- parent exact rotary: 2,500 no-grad, 1,200 grad/replay forward, 600 backward, zero reference-grad calls;
- no CUDA/cgroup OOM; rank-0 peak allocated/reserved 34.79/36.65 GiB;
- all 16 GPUs idle and all three atomic locks free after teardown.

This is one unprofiled run. The measured improvement promotes the bounded mechanism only; it is not a median or variance-qualified speedup.

Persistent raw evidence:

- `iteration_381_qk_rmsnorm_rotary_canaries_20260826`
- `iteration_382_qk_rmsnorm_rotary_world16_20260826`
- `iteration_383_qk_rmsnorm_rotary_world16_r2_20260826`
