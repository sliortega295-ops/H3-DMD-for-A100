# H3 DMD grad/replay LoRA epilogue (2026-08-26)

The default-off `H3_LORA_GRAD_EPILOGUE` candidate extends the already validated
BF16 no-grad LoRA-B residual epilogue with an exact custom-autograd path. It
does not change the LoRA GEMMs, workload schedule, FSDP2, attention, activation
policy, or training math.

| Run | Unprofiled cycle | Peak allocated | Status |
|---|---:|---:|---|
| Iteration 388 parent | 816.527 s | 34.77 GiB | PASS |
| Iteration 390 grad LoRA epilogue | **813.609 s** | **33.39 GiB** | PASS, provisional single run |

The single-variable delta is `-2.918 s` (`-0.357%`, `1.00359x`) and peak
allocated HBM fell by 1.38 GiB. This is one matched run, not a
variance-qualified speed claim.

## Evidence

- Real installed-PEFT BF16 production-shape output and input/LoRA-A/LoRA-B
  gradients were bitwise equal to the reference.
- A two-rank FSDP2 live-weight canary passed with identical cross-rank output
  and input-gradient hashes.
- The world16 run passed 24/6/1 application forwards, 1/5/0 grad forwards,
  1/5 backwards, 96 unique samples, and Grid replay 300/600/600/0.
- Per-rank LoRA receipts were 7488 fused no-grad entries, 3744 fused
  grad/replay entries, 1872 actual custom backward executions, and zero
  reference-grad or invalid-contract calls.
- Both nodes exited 0 with zero OOM delta and clean teardown.

Large logs remain in the server evidence directories for Iterations 389 and
390; this repository stores only the compact result.
