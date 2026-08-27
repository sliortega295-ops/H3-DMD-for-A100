# Iteration 404 — bounded exact CPU-staged compact FA3 replay cache

## Outcome

**FAIL_REPEATED_SAFETY_HBM; NOT A TIMING RESULT.**

The one-variable recovery from Iteration 403 limited asynchronous D2H to two entries and reused one event-ordered pinned host-buffer pool. Unit tests and the real one-GPU integration canary passed bitwise, but the real world16 run reproduced the same HBM safety failure at essentially the same phase and magnitude.

| Run | D2H policy | node0 min free | node1 min free | CUDA/cgroup OOM | Timing |
|---|---|---:|---:|---:|---:|
| Iteration 403 | unbounded async | 710 MiB | 1,502 MiB | 0 / 0 | NOT_RUN |
| Iteration 404 | max in-flight = 2 | 722 MiB | 1,150 MiB | 0 / 0 | NOT_RUN |

Node0 triggered the frozen 1 GiB safety floor and its owned process group was stopped. The peer was subsequently cancelled. All GPUs and all three atomic locks are clean.

## Evidence boundary

- exact source: `c2c123a43f0ea4380dd617557db0c50486ec05a5`
- focused tests: 5/5 PASS
- full tests: 71/71 PASS
- real CUDA integration canary: output and input gradient bitwise PASS; D2H/H2D bytes reconcile exactly
- node0 max driver use: 39,616 MiB; min free: 722 MiB
- node1 max driver use: 39,188 MiB; min free: 1,150 MiB
- node0 cgroup peak: 499,602,288,640 B
- node1 cgroup peak: 479,621,033,984 B
- `oom=0`, `oom_kill=0` on both nodes

The almost identical peak time and magnitude falsify the claim that an unbounded D2H queue was the dominant HBM source. The remaining peak is associated with the compact custom-autograd checkpoint/replay path plus the existing FSDP/FA3 transient working set. Exact ownership below that boundary was not measured, so it remains `UNRESOLVED_COMPACT_REPLAY_TRANSIENT`; it is not called model offload.

## Decision

Do not run depth=1 or increasingly invasive replay-cache patches automatically. This mechanism reproduced the documented failure after its one bounded repair. Return to the validated 812.029 s substrate. The 780 s target was not reached. A further run requires a new preregistered mechanism or explicit authorization to change the 1 GiB safety contract.
