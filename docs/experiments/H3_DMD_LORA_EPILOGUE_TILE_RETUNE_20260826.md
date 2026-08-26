# Iteration 396: BF16 LoRA epilogue tile retune

## Result

The exact world16 workload completed, but an artifact-local shell postcheck
returned nonzero after timing. Therefore this is a **directionally positive
diagnostic**, not a formally promoted current best.

| Run | LoRA epilogue `BLOCK_N` | One-cycle wall | Peak allocated | Status |
|---|---:|---:|---:|---|
| Iteration 393 parent | 64 | 812.029 s | 33.39 GiB | PASS |
| Iteration 396 | 128 | 810.336 s | 33.39 GiB | PASS workload; postcheck exit 144 |

Observed delta: **-1.693 s (-0.208%)**, or 1.00209x in one unprofiled run.
This magnitude is within run variance and is not a formal speedup claim.

## Change and microbenchmark

Only the existing exact BF16 LoRA-B GEMM plus residual epilogue tile changed
from `m64n64k32w8s3` to `m64n128k32w8s3`. The selector is default-off and
fail-closed. Thirty alternating pairs at the three production projection widths
were bitwise-equal and projected 2.311 s/GPU/cycle kernel-sum reduction.

Installed-PEFT forward/backward and live 2-rank FSDP2 BF16 canaries passed.

## Workload gates

- Exact measured source: `da34365c001899149f0e9c68609124dab75f0d60`.
- 16/16 ranks and 96 rank-qualified sample identities.
- Application census: 24/6/1 forwards, 1/5/0 grad-forwards, 1/5 backwards.
- Grid replay: 300 wraps, 600 scopes, 600 hits, 0 misses.
- LoRA epilogue: 7,488 calls, 3,744 grad-enabled, 1,872 checkpoint-replay,
  zero fallback.
- No CUDA/cgroup OOM; all GPUs and locks were clean after teardown.

## Evidence boundary

Node0 returned `144` only because its post-run grep expected
`epilogue_block_n` in the cycle log. The measured source exposed the field in
the registration receipt and the immutable preflight recorded the active value,
but that source did not print it in the cycle log. A follow-up evidence-only
commit fixed the logger; the measured workload was not rerun. Raw logs and
receipts remain under the external research-artifact directory.

