# Codex handoff: controlled 16×A100 MiniMax-H3 DMD run

Execute this task on the real 2-node × 8-A100-40GB cluster. Do not stop at static review: run the experiments and preserve all failures/results.

## 0. Exact source to run

Repository:

```text
https://github.com/sliortega295-ops/H3-DMD-for-A100
```

Branch:

```text
agent/h3-a100-shared-backbone
```

Before doing any GPU work, record the exact branch HEAD and refuse to run an older checkout. The controlled primary config is:

```text
configs/minimax_h3_t2av_dmd_a100_world16.yaml
```

Do not use historical segment=8 OOM evidence as a search prior for this task. Diagnose only from the new controlled shared-backbone run.

## 1. Control variables are fail-closed

The primary run must retain:

```text
hardware              2 nodes x 8 A100-40GB
world size            16
placement             world16 FSDP2
sequence parallel     1
resolution            768 x 1344
frames                 124
model dtype            BF16
batch/rank             1
global batch           16
Student LoRA           rank 128
Fake LoRA              rank 128
num_inference_steps    4
fixed end_step_idx     3
Student updates        1 / outer cycle
Fake updates           5 / outer cycle
renoise sigma          continuous random [0.02, 0.98]
attention backend      _flash_3_hub
```

Every outer cycle must pass the repository's matched census on all ranks:

```text
Student application forwards       24
Fake application forwards            6
Teacher application forwards         1
Student grad-enabled forwards         1
Fake grad-enabled forwards            5
Teacher grad-enabled forwards         0
Student backward graphs               1
Fake backward graphs                  5
```

Across world16, Student + Fake0..4 must produce 96 distinct rank-qualified sample identities per cycle. Gradient-checkpoint recomputation is not an extra application forward.

If any census/sample check fails, fix the control-variable bug before doing performance work. Never report a speedup from such a run.

## 2. Static/source gate

On both nodes:

1. checkout the exact branch HEAD;
2. run `bash scripts/bootstrap_lightx2v.sh`;
3. run the repository unit/static checks;
4. run `scripts/preflight.py`.

The preflight must confirm the pinned LightX2V, Diffusers source, `_flash_3_hub`, model layout and 8 visible A100s per node.

Infrastructure/source/shared-filesystem/NCCL failures are not OOMs. Classify them separately.

## 3. One-node smoke

Run the 8-GPU smoke first. It must complete Student1 + Fake5 and pass the 8-rank matched census/48-sample identity gate.

The smoke is a control-flow gate only; it is not a full-resolution memory result.

## 4. Primary world16 capacity/correctness gate

Set:

```bash
export H3_MAX_ITERS=1
export H3_SAVE_EVERY=0
export H3_ATTN_BACKEND=_flash_3_hub
export H3_BENCHMARK_SEED=42
```

Launch the default `scripts/launch_2node.sh` on both nodes.

Record on every rank at setup, Student forward/backward, each Fake forward/backward/update and failure:

- CUDA allocated/reserved/max allocated/max reserved;
- `cudaMemGetInfo` free/total;
- host RSS;
- cgroup memory.current/peak and oom/oom_kill deltas;
- finite Student/Fake losses;
- matched census;
- Student/Fake adapter/optimizer parameter separation.

Required success is a complete exact Student1 + Fake5 outer cycle with no OOM and all matched gates passing.

## 5. Timing boundary

Do not include checkout, checkpoint/model loading, dataset construction, FSDP construction or preflight in training timing.

Add low-overhead timing receipts if the current branch does not yet expose them. Do not alter the application compute graph to add timing.

For one-cycle wall:

```text
world16 barrier
CUDA synchronize
START
Student update
5 Student no-grad Fake-stage rollouts
Fake forward/backward/optimizer x5 in order
CUDA synchronize
world16 barrier
STOP
```

Stop the timer **before** the matched-census `all_gather_object` / result serialization so correctness bookkeeping is not counted as training work.

For FULL5 wall, run five consecutive exact outer cycles. The formal FULL5 timer begins before cycle 1 Student and ends after cycle 5 Fake5 commit, with CUDA/world16 synchronization only at the outer boundaries. Do not put correctness-gather/reporting work inside the FULL5 timed window; collect local census snapshots and validate them after the timed region.

Also report five individual outer-cycle training times using CUDA events/NVTX or equivalent low-overhead timestamps without inserting additional global barriers between cycles.

## 6. Formal timing

After the one-cycle capacity gate passes, run:

```bash
export H3_MAX_ITERS=5
export H3_SAVE_EVERY=0
```

Run at least three independent **unprofiled** FULL5 trials with the same source/config/assets. Report all raw values, median, mean, CV and peak HBM/host memory. Do not pick the best run.

At minimum break down:

- Student total;
- Student rollout;
- Fake/Teacher score during Student loss;
- Student backward;
- Student optimizer;
- five Student no-grad rollout preparation total;
- Fake1..5 forward/backward/optimizer;
- total Fake5 phase;
- FULL5 wall.

## 7. If the exact primary run CUDA-OOMs

Use only the new run's memory evidence. Change **one mechanism at a time** and keep every control variable above fixed.

Suggested order:

### C1: Selective Activation Checkpointing / checkpoint policy

Use PyTorch selective activation checkpointing if compatible with the pinned stack. Prefer recomputing large/cheap intermediates and retaining only expensive outputs that fit the budget. Preserve the same 31 application forwards and 6 backward graphs; checkpoint recomputation is internal and must not change the DMD objective.

### C2: Selective activation offload

If C1 is insufficient, offload only large saved tensors on grad-enabled Student/Fake backward paths. Start with a high threshold such as 128 MiB; lower to 64 MiB only if needed. Use pinned memory/asynchronous copies when safe. Do not activation-offload no-grad Student rollouts or Teacher score work without evidence it is needed.

Record H2D/D2H bytes, offloaded tensor count, HBM reduction and wall-time penalty.

### C3: CP/SP=2

If full-sequence activation remains dominant, implement/reuse H3 context/sequence parallelism of size 2 while preserving global attention, packed video/audio/text ordering, RoPE/timestep/tag semantics, B1 per logical data rank and identical loss/update semantics.

Prefer communication groups inside the node/NVLink domain. Validate numerical equivalence before timing.

### Host/cgroup OOM during cold construction

This is not solved by changing the workload. Implement meta/direct sharded safetensor loading or another equivalent host-memory-safe initialization path. Cold construction remains outside the formal training timer.

Do **not** use the following to claim the exact workload fits:

- lower resolution/frames;
- lower LoRA rank;
- fewer rollout evaluations or Fake updates;
- different batch/update normalization;
- INT8/FP8 model weights;
- sigma discretization;
- merging five Fake optimizer steps into one batch update.

Sigma-grid + permanent AdaLN weight dropping is a separate later ablation, not an exact-baseline rescue path.

## 8. HSDP placement ablation

Only after a valid primary world16 result exists, optionally run:

```text
configs/minimax_h3_t2av_dmd_a100_2x8.yaml
```

This is a systems placement ablation. It must pass the exact same world16 census and 96-sample contract. Do not broadcast one sample/noise/sigma across an 8-way HSDP shard group.

## 9. Correctness evidence

For controlled variants, preserve:

- source/config/model/cache/seed identities;
- per-rank sample sequence;
- application forward/grad-forward/backward census;
- Student/Fake LoRA gradient norms;
- final Student/Fake local parameter-shard hashes;
- final Student/Fake Adam `exp_avg` / `exp_avg_sq` shard hashes where practical;
- Student/Fake losses.

When comparing a schedule-only variant against its same-placement control, refuse to emit speedup if sample/census/state equivalence gates fail.

## 10. Final artifacts

Write:

```text
docs/experiments/H3_DMD_A100_CONTROLLED_FULL5.md
artifacts/h3_dmd_a100_controlled_full5.json
```

The result table must include:

```text
variant
source HEAD
placement
memory mechanism
matched census PASS/FAIL
sample contract PASS/FAIL
capacity status / failure location
peak allocated/reserved/free HBM
host/cgroup peak
one-cycle training wall
FULL5 run1/run2/run3
FULL5 median/CV
Student wall
Fake5 wall
H2D/D2H if applicable
correctness/state status
```

Choose the fastest candidate only among variants that preserve the exact contract and complete the full five-cycle workload stably.

After formal unprofiled timing, capture one representative Nsight Systems run for attribution only. Nsight/profiled wall is not the formal denominator.
