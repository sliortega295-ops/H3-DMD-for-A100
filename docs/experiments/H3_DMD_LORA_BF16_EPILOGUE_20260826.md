# Iteration 376 — live-BF16 no-grad LoRA epilogue

Status: **PASS / provisional single-run improvement**.

- Exact HEAD: `bf504375c60c621507e28744afcc29725eb0432b`.
- One matched, unprofiled world16 `Student1 + Fake5` cycle: **828.355 s**.
- Direct valid parent Iteration 367: **836.298 s**.
- Single-variable delta: **-7.943 s** (-0.950%), speedup **1.0096x**.
- Cumulative vs Iteration 327 Grid1000 control (904.986 s): **-76.631 s** (-8.468%), speedup **1.0925x**.

The first production attempt failed closed before timing because the earlier microbenchmark assumed FP32 LoRA operands. A bounded world16 dtype audit then proved the actual sharded workload uses BF16 for the LoRA-A output, LoRA-B weight, and base output. The corrected kernel fuses the BF16 LoRA-B GEMM and residual add while explicitly preserving eager's intermediate BF16 rounding point.

Correctness evidence:

- deterministic nonzero installed-PEFT CUDA canary: bitwise output parity and bitwise fixed-adjoint input/parameter gradients; median **14.859 -> 13.449 ms** for a production-size layer;
- 2-rank FSDP2 canary: both ranks emitted the same output SHA with BF16 LoRA A/B;
- world16 all-gathered contract: 16/16 rank snapshots, 96 unique sample identities, application forwards 24/6/1, grad forwards 1/5/0, backwards 1/5;
- Grid replay 300/600/600/0; no invalid LoRA calls;
- 7,488 no-grad epilogues fused, 3,744 grad/replay epilogues left on the validated reference path, and 312 Teacher-disabled calls left untouched.

Rank0 peak allocated/reserved remained **34.79/36.63 GiB**. Both nodes exited 0, cgroup OOM deltas were zero, all 16 GPUs were clean, and all three launch locks were free after teardown.

This is one matched unprofiled run, not a confidence interval. Nsight was not used for the timing number.
