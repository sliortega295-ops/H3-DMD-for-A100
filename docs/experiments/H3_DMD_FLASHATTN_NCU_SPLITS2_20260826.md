# H3 Grid1000 FlashAttention-3 NCU and no-grad split result

## Scope

This evidence keeps the matched world16 Grid1000 workload fixed and diagnoses
the pinned `_flash_3_hub` kernel at the production attention shape
`[1,37760,56,128]`. Nsight Compute durations are diagnostic only; the world16
performance number is from an unprofiled run.

## Kernel evidence

The native forward microbenchmark is 185.127 ms, within 0.1% of the trusted
world16 Nsight Systems mean, so the captured shape is representative. The main
forward kernel reports 73.51% compute/SM throughput, 73.8% tensor-pipeline
activity, only 1.50% DRAM throughput, 12.47% achieved occupancy, 255 registers
per thread, and 65.92 KiB dynamic shared memory per block. The main backward
kernel reports 66.09% compute throughput, 66.3% tensor-pipeline activity, 1.98%
DRAM throughput, 12.5% occupancy, 255 registers per thread, and 165.12 KiB
shared memory. The backward kernel is additionally L1/TEX constrained (72.89%).
These are not MFU measurements.

The bounded `num_splits` sweep found `2` faster than `1`, while `4` was neutral
and `8` regressed. Ten alternating real-shape pairs measured medians of
185.443 ms and 183.527 ms (`-1.753 ms`). Direct outputs were close but not
bitwise equal: max-abs 2.44140625e-4, normalized-L2 0.0022124, cosine
0.99999756. A custom FA3 source rewrite was therefore rejected; only the
existing scheduling argument was tested.

## World16 result

Iteration 365 enabled `num_splits=2` only when autograd was disabled. The
runtime receipt proves that exactly 1,300 no-grad attention calls were rewritten
and all 624 grad/checkpoint-replay calls remained at `num_splits=1`.

| Run | One cycle | Delta | Status |
|---|---:|---:|---|
| Iteration 362 parent | 849.808 s | -- | PASS |
| Iteration 365 no-grad split=2 | 848.128 s | -1.680 s (-0.198%) | PASS, one run |

All 16 ranks completed; application forward census was 24/6/1, grad forward
1/5/0, backward 1/5, and Grid replay 300/600/600/0. Peak rank0 HBM remained
34.79 GiB allocated / 36.65 GiB reserved, with no CUDA or cgroup OOM and clean
teardown.

This is a provisional one-run improvement, not a confidence interval. The
remaining FA3 opportunity is primarily an upstream kernel resource/latency
problem rather than a DRAM bottleneck; a repository-local kernel rewrite is not
justified by the bounded evidence.
