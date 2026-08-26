# Iteration 397: LoRA epilogue local refinement

The one-GPU production-shape sweep found `m64n128k32w4s4g8` faster than the
Iteration 396 `m64n128k32w8s3g8` launch at all three widths, with bitwise-equal
outputs. Its weighted incremental projection was only **0.592 s/GPU/cycle**,
below the pre-registered 1.0-second threshold, so no source candidate or
world16 run was started.

| Shape | Existing tile | Local winner | Change |
|---|---:|---:|---:|
| 37,760 × 7,168 | 0.964608 ms | 0.923648 ms | -4.246% |
| 37,760 × 5,376 | 0.736256 ms | 0.709632 ms | -3.616% |
| 37,760 × 28,672 | 3.970048 ms | 3.817472 ms | -3.843% |

Status: `COMPLETE_NO_GO_WORLD16`. This is microbenchmark evidence, not a
training speedup.

