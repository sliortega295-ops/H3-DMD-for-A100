# Iteration 362 — exact no-grad SwiGLU result

Status: **PASS / provisional single-run improvement**.

- Exact HEAD: `60a4a73770e48a6837ac8857cd44c28616963540`.
- One matched, unprofiled world16 outer cycle: **849.808 s**.
- Direct parent Iteration 359: **853.376 s**.
- Single-variable delta: **-3.568 s** (-0.418%), speedup **1.0042x**.
- Cumulative vs Iteration 327 control: **-55.178 s** (-6.097%), speedup **1.0649x**.
- Formal switch: `H3_FUSED_SWIGLU=1`, `H3_FUSED_SWIGLU_GRAD=0`. Projection GEMMs and all 600 grad/replay calls remain on the pinned Diffusers reference path.
- Real-shape canary (37,760 x 14,336): output bitwise equal; activation kernel **6.212 ms -> 2.345 ms (2.649x)**. The experimental fused backward was deliberately excluded because its local input gradient was not bitwise equal (max abs 0.001953125, nL2 1.19e-6).
- World16 collective gate: application forwards 24/6/1, grad forwards 1/5/0, backwards 1/5; Grid replay 300/600/600/0.
- SwiGLU receipt: 1,250 fused no-grad calls, 600 reference grad/replay calls, zero fused backward calls.
- No boundary offload; no CUDA/cgroup OOM; rank0 peak allocated/reserved **34.79/36.65 GiB**; both nodes exited 0 and GPUs were clean after teardown.
- The observed ~3.6 s improvement matches the trusted Iteration 360 per-rank SiLU/multiply kernel budget. It is still one run, not a confidence interval.
