# Iteration 384 — fresh profile of the current H3 winner

## Identity and validity

- Profile run: `iteration_384_qk_rmsnorm_rotary_nsys2025_20260826`.
- Exact profile HEAD: `cc0e1b5bab240e8a99699516fef255f36ddcec4c`.
- Workload result HEAD: `bab7b3b6c240ff3410021bceb500f67a3bdf6011`; runtime candidate: `0f596ca82f4e547b3046f1ccf251f82ae9d3c662`.
- Frozen formal unprofiled result: **823.037 s**. The profiled range is diagnostic only.
- Nsight Systems `2025.5.1.121`, `nsys-ai 0.3.0`; profiler overhead health check `0.4%`.
- Both nodes have 8/8 devices, 16/16 exact-range receipts, valid SQLite exports and zero unresolved-profile errors. GPU-metric coverage is at least 99.982% for every reported phase/metric.
- Matched contracts stayed at application forwards 24/6/1, grad forwards 1/5/0, backward 1/5, samples 96, and Grid replay 300/600/600/0. CUDA/cgroup OOM delta is zero.

## Overlap-aware full-cycle result

All values below are the 16-rank median of per-device timelines.

| Category | Union / observed | Hidden under non-NCCL compute | Not covered by non-NCCL compute | Share of profile wall |
|---|---:|---:|---:|---:|
| Profile wall | 823.657 s | -- | -- | 100% |
| Non-NCCL kernel coverage | 811.335 s | -- | -- | 98.50% |
| NCCL | 188.785 s | 182.528 s | 6.251 s | 0.76% exposed |
| CUDA memcpy | 0.096 s | ~0.000 s | 0.096 s | 0.012% exposed |
| No traced GPU activity | 6.054 s | -- | -- | 0.735% |

The NCCL hidden fraction is about 96.7%. H2D is only about 0.233 GiB/rank and D2H about 0.00019 GiB/rank. There is no remaining activation/model-offload traffic worth optimizing in this path.

## Actual sampled hardware metrics

| Phase | Wall | SM Active | SM Issue | Tensor Active | DRAM read | DRAM write |
|---|---:|---:|---:|---:|---:|---:|
| Full cycle | 823.657 s | 97.78% | 27.11% | 71.53% | 11.16% | 3.79% |
| Student | 153.643 s | 95.25% | 26.54% | 70.12% | 10.88% | 3.64% |
| 5x Generator rollout | 329.659 s | 98.42% | 29.03% | 75.71% | 11.63% | 3.68% |
| Fake updates (typical) | 62.7–69.5 s | 98.15–98.49% | 25.49–25.61% | 68.09–68.33% | ~10.7% | ~3.9% |

This is compute/tensor-pipeline dominated, not DRAM-bandwidth dominated. `SM Active` and `Tensor Active` are hardware samples; non-NCCL kernel coverage is not MFU.

## Kernel evidence and next bounded candidate

Per-rank median kernel-duration sums are dominated by FlashAttention forward (354.814 s), BF16 GEMM (251.266 s), and FlashAttention backward (152.764 s). Those kernels already have the prior NCU evidence and are not safe repository-local rewrites.

The remaining bounded exact pointwise boundary is grad/checkpoint-replay Q/K RMSNorm immediately followed by exact rotary. The fresh profile still records per-rank layer-norm forward/backward sums of **3.903/2.518 s**, plus standalone grad rotary kernels. The next experiment will fuse only that pinned Q/K chain, retain the frozen attention/DiT count and use a custom backward with production-shape parity before any world16 run.

## nsys-ai interpretation guard

`nsys-ai` correctly classifies the busiest-device critical path as 97.87% GPU-compute, 1.37% exposed communication, and 0.76% no-GPU activity. Its generic stream-gap finding reports 65% "idle" by adding gaps on auxiliary streams; that is not device idle and is rejected here. The mutually exclusive per-device partition above reconciles each phase wall and is the claimable denominator.

Timeline overlay: `http://127.0.0.1:8384` while the local viewer process is running.

## Claim boundary

- Profiled time is diagnostic only and does not replace 823.037 s.
- NCCL exposed means not temporally covered by a non-NCCL CUDA kernel; it is not guaranteed removable time.
- Kernel duration sums rank candidates but are not additive wall-clock overhead.
