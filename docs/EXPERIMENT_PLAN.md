# Validation and ablation plan

The first goal is a **controlled world16 MiniMax comparison**, not an unconstrained speedup search. Formal timing is invalid unless model/data/loss/update semantics and application-level DiT counts match the DMD-System reference.

## A. Matched-compute gate

Primary config:

```text
configs/minimax_h3_t2av_dmd_a100_world16.yaml
```

Per outer cycle and per rank, require exactly:

```text
Student DiT forwards          24
Fake DiT forwards              6
Teacher DiT forwards           1
Student grad-enabled forwards  1
Fake grad-enabled forwards     5
Teacher grad-enabled forwards  0
Student backward graphs        1
Fake backward graphs           5
fixed end_step_idx              3
```

Across world16, the six stage samples must yield 96 distinct rank-qualified identities. If any census differs, do not report wall time as a matched result.

## B. Source/hardware preflight

Controlled runs pin the same LightX2V/attention environment as the DMD-System reference. Run `scripts/preflight.py` on both nodes before model construction. Treat launcher/source/shared-filesystem failures separately from CUDA/host OOM.

## C. One-node smoke

Use the smoke canvas for one Student1/Fake5 cycle. The smoke must pass the same per-rank `24/6/1` forward and `1/5` backward census before any full-resolution run.

## D. Full-resolution world16 capacity gate

Run one exact outer cycle at 1344×768×124 with world16 FSDP2. Record peak allocated/reserved HBM, driver free memory, host/cgroup memory, finite losses, adapter correctness and matched census.

Do not reduce resolution, frames, LoRA rank, Fake update count, rollout evaluations, or precision to make it fit.

## E. Five-cycle formal timing

After capacity passes, run five consecutive outer cycles with checkpoint/inference side effects disabled. Cold checkout/model load/FSDP construction are excluded. Synchronize world16/CUDA at the timing boundaries and report both per-cycle and five-cycle wall time.

Run at least three independent unprofiled repeats for the final winning exact configuration; report all raw values, median and dispersion.

## F. Systems ablations

Change one systems dimension at a time while preserving the matched census.

| ID | Placement | Shared backbone | AdaLN cache | 5G→5F reorder | Purpose |
|---|---|---:|---:|---:|---|
| P0 | world16 FSDP2 | yes | on | on | primary controlled candidate |
| P1 | world16 FSDP2 | yes | off | on | isolate AdaLN cache |
| P2 | world16 FSDP2 | yes | on | off | isolate phase clustering |
| P3 | 2×8 HSDP | yes | on | on | placement/communication ablation |

HSDP must retain 16 independent global-rank samples/RNG streams; it is not allowed to collapse global batch16 into two node-level samples.

## G. OOM follow-up

If the exact primary candidate OOMs, preserve the failure evidence and change one memory mechanism per retry. Prefer memory mechanisms that do not alter the DMD objective: checkpoint/SAC policy, selective activation offload, then context/sequence parallelism if activation remains dominant. Keep continuous renoise sigma for the exact path.

A finite sigma grid plus permanent AdaLN weight dropping is a separate algorithmic/memory ablation, not the exact matched baseline.

## H. Profiling

Only after unprofiled timing exists, capture one representative Nsight run. Report compute/NCCL/H2D/D2H interval unions independently because they can overlap. Never substitute profiled wall time for the formal denominator.
