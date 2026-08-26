# Iteration 379 — no-grad RMSNorm + modulation world16

- Status: `PASS_PROVISIONAL_SINGLE_RUN`
- Source: `7a9e0e3da0eb477a22e29287a7b618b50e260f6f`
- Frozen parent: Iteration 376, 828.355 s
- Candidate: **825.947 s**
- Delta: **-2.408 s (-0.291%)**, provisional 1.002915x
- Contract: world16; application 24/6/1, grad 1/5/0, backward 1/5; 96 rank-qualified sample identities
- Feature census: 2,500 no-grad RMSNorm+modulation calls/rank
- Grid replay: 300 wraps, 600 scopes, 600 hits, 0 misses
- Peak allocated/reserved rank0: 34.79/36.65 GiB
- Node exits: 0/0; CUDA/cgroup OOM delta: 0; teardown: all 16 GPUs idle and all locks free

This is one unprofiled trial, not a three-run median. It promotes the mechanism as a bounded improvement only; it does not establish a variance-qualified speedup.
