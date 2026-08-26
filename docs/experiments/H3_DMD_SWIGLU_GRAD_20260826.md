# H3 DMD grad/replay SwiGLU fusion (2026-08-26)

## Result

The bounded world16 test enabled only `H3_FUSED_SWIGLU_GRAD=1` on top of the
Iteration 386 stack.

| Run | Unprofiled cycle | Delta | Status |
|---|---:|---:|---|
| Iteration 386 parent | 820.008 s | — | PASS |
| Iteration 388 grad/replay SwiGLU | 816.527 s | -3.481 s (-0.425%) | PASS, provisional single run |

This is a one-run mechanism check, not a variance-qualified speed claim.

## Correctness and capacity

- world16, 24/6/1 application forwards, 1/5/0 grad forwards, and 1/5
  backwards all passed;
- Grid1000 replay was 300 wraps, 600 scoped executions, 600 hits, 0 misses;
- per rank SwiGLU census was 1250 no-grad entries, 600 grad/replay entries,
  300 custom-autograd backward calls, and 0 reference grad entries;
- rank0 peak allocated HBM was 34.77 GiB; both nodes had zero OOM delta;
- all 16 GPUs and atomic locks were clean after the run.

## Receipt note

The training command completed and its in-process source contract passed.  The
artifact-local node0 wrapper subsequently returned 109 because it expected 600
SwiGLU backward calls.  The source correctly pins 300: six grad graphs times 50
blocks.  A corrected post-hoc verifier passed the immutable train log, so the
GPU run was not repeated.

Full logs and immutable evidence remain under
`iteration_388_swiglu_grad_world16_20260826` on the experiment server.
