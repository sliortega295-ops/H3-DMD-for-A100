# Codex handoff: shared-backbone H3 on 16×A100 with proven boundary staging

Execute this task on the real 2-node × 8-A100-40GB cluster. Do not stop at static review: run the experiments and preserve all failures/results.

## 0. Source and reference

Repository:

```text
https://github.com/sliortega295-ops/H3-DMD-for-A100
```

Branch:

```text
agent/h3-a100-shared-backbone
```

Before GPU work, fetch the branch and record the exact current HEAD. Do not reset to an older SHA from a previous prompt.

Primary config:

```text
configs/minimax_h3_t2av_dmd_a100_world16.yaml
```

The memory-policy reference is the DMD-System branch that already completed the same MiniMax-H3 world16 one-cycle workload:

```text
repo:   sliortega295-ops/DMD-System
branch: agent/lightx2v-minimax-selective-boundary-offload-20260818
```

Its preserved one-cycle evidence reports:

```text
Native control              PASS   917.993973 s   max allocated 27.94 GiB
DistillSchedule phase       PASS   914.102536 s   max allocated 28.10 GiB
```

Both used native per-block checkpointing plus checkpoint-boundary CPU staging and completed 31 application forwards / 6 backward graphs on all 16 ranks.

Our first goal is therefore NOT to invent another memory mechanism. First verify that the shared-backbone version reproduces this proven activation policy and should fit with less static model residency.

## 1. Frozen compute contract

Primary run must retain:

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
benchmark seed         20260817
```

Every outer cycle must pass:

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

Across world16, Student + Fake0..4 must produce 96 distinct rank-qualified sample identities per cycle.

## 2. Required activation policy

The PRIMARY path is now exactly:

```text
activation checkpoint segment = 1
(native per-block checkpointing)

activation policy = checkpoint_boundary_cpu

for each grad-enabled H3 DiT forward:
    50 transformer blocks
    -> 50 native checkpoint calls
    -> save only each checkpoint's first input hidden_state to pinned CPU
    -> close saved-tensor hook when that checkpoint call returns
    -> backward recompute runs normally on GPU
```

Do NOT set the old experiment variables:

```text
H3_ACTIVATION_CHECKPOINT_SEGMENT=8
H3_ACTIVATION_OFFLOAD=1
H3_ACTIVATION_OFFLOAD_MIN_BYTES=...
```

Before launching, explicitly unset them:

```bash
unset H3_ACTIVATION_CHECKPOINT_SEGMENT
unset H3_ACTIVATION_OFFLOAD
unset H3_ACTIVATION_OFFLOAD_MIN_BYTES
```

The config should resolve to:

```text
segment_size = 1
activation_policy.name = checkpoint_boundary_cpu
pin_memory = true
```

Formal timing should keep `H3_BOUNDARY_OFFLOAD_EVENTS=0`; aggregate counters are still checked by the trainer. Diagnostic JSONL events may be enabled only in a separate debugging run.

## 3. Boundary-policy fail-closed contract

Because the shared physical backbone performs one Student grad DiT and five Fake grad DiTs per cycle, every passing outer cycle must additionally observe per rank:

```text
grad transformer forwards      6
Student grad forwards          1
Fake grad forwards             5
native checkpoint calls      300
checkpoint boundary CPU copies 300
```

That is 6 × 50 checkpoint boundaries.

If the trainer does not report exactly these values, the memory policy was not actually applied and the run is invalid even if it does not OOM.

Checkpoint replay is outside the save hook. Do not recursively CPU-offload FFN/attention/LoRA recompute intermediates.

## 4. Static/source gate

On both nodes:

1. update to the exact current branch HEAD;
2. `bash scripts/bootstrap_lightx2v.sh`;
3. `bash scripts/run_unit_tests.sh`;
4. run `scripts/preflight.py`;
5. confirm `_flash_3_hub`, pinned LightX2V/Diffusers source, model/cache identity and 8 visible A100s per node.

Infrastructure/source/shared-filesystem/NCCL failures are not OOMs.

## 5. One-node smoke

Run the 8-GPU smoke first. It now exercises the same checkpoint-boundary policy.

Required:

```text
Student1 + Fake5 complete
matched forward/backward census PASS
boundary-offload 6 / 1 / 5 / 300 / 300 counters PASS
no CUDA OOM
no cgroup OOM
```

Smoke timing is diagnostic only.

## 6. Primary world16 one-cycle gate

Use:

```bash
export H3_MAX_ITERS=1
export H3_SAVE_EVERY=0
export H3_ATTN_BACKEND=_flash_3_hub
export H3_BENCHMARK_SEED=20260817
export H3_BOUNDARY_OFFLOAD_EVENTS=0
unset H3_ACTIVATION_CHECKPOINT_SEGMENT
unset H3_ACTIVATION_OFFLOAD
unset H3_ACTIVATION_OFFLOAD_MIN_BYTES
```

Launch `scripts/launch_2node.sh` on both nodes.

Record on every rank:

- CUDA allocated/reserved/max allocated/max reserved;
- `cudaMemGetInfo` free/total;
- host RSS and cgroup memory peak/OOM deltas;
- finite Student/Fake losses;
- matched census;
- boundary-offload counters;
- Student/Fake adapter separation.

The previous shared-backbone failures reached about 37.87 GiB allocated because they used segment=8 plus thresholded saved-tensor offload. Do not compare that failed policy against the new run as if the activation policy were unchanged.

The DMD-System all-boundary control passed at about 27.94 GiB max allocated despite three physical model roles. Therefore a shared-backbone run that correctly applies the same boundary policy but still peaks far above ~28 GiB requires direct memory attribution before trying CP/SAC.

## 7. If the new boundary-policy run still OOMs

Do not immediately add another mechanism. First run a matched memory comparison against DMD-System and identify the unexplained residency delta.

Compare at these points:

```text
after setup
before Student grad DiT
end Student grad forward
first Student backward recompute
FFN LoRA projection at failure
end Student backward
```

Attribute per rank:

```text
FSDP parameter shard bytes
currently materialized full FSDP units
Student/Fake LoRA parameter bytes
Student/Fake optimizer-state bytes
AdaLN persistent/dynamic cache bytes
checkpoint boundary saved bytes on CPU
GPU saved-tensor live bytes
five prepared Fake rollout latent bytes
allocator inactive/reserved bytes
NCCL/FSDP buffers
```

The question to answer is:

> Why is shared-backbone peak HBM not lower than the DMD-System 27.94 GiB control under the same checkpoint-boundary policy?

Only after this attribution should you change another memory mechanism.

Likely next checks, in order:

1. verify the shared model's FSDP unit/materialization lifetime matches upstream;
2. verify both named LoRA adapters are sharded and inactive adapter weights/states are not materialized full-size;
3. verify AdaLN caching does not keep FSDP full parameters resident;
4. verify phase-cluster prepared rollouts are only latent tensors and release promptly;
5. only then consider chunked/fused LoRA residual or CP/SP2 if live operator workspace is still the blocker.

Do not return to segment=8 unless new evidence specifically justifies it.

## 8. Timing after capacity PASS

Cold setup is excluded.

One-cycle training timer:

```text
world16 barrier
CUDA synchronize
START
Student update
5 Student no-grad rollouts
5 ordered Fake forward/backward/optimizer commits
CUDA synchronize
world16 barrier
STOP
```

Keep correctness `all_gather_object`, JSON serialization and detailed boundary event logging outside formal timing.

After one-cycle PASS, run five consecutive cycles and at least three independent unprofiled FULL5 trials. Report all raw FULL5 values, median, mean, CV and peak HBM/host memory.

## 9. Compare against DMD-System

Once shared-backbone one-cycle capacity passes, report side-by-side:

```text
DMD-System Native boundary policy: 27.94 GiB, 917.993973 s one cycle
DMD-System phase-cluster:           28.10 GiB, 914.102536 s one cycle
H3 shared-backbone:                 <measured>
```

Do not claim schedule speedup merely from one run. The immediate question is memory and capacity: shared backbone should reduce static model residency while preserving the same compute and activation policy.

For schedule correctness, retain the existing state/sample/hash gates. The DMD-System phase-cluster evidence had a Fake-state parity failure despite capacity PASS, so do not use its ~0.4% one-run timing delta as a formal speedup claim.

## 10. HSDP ablation

Only after world16 FSDP2 passes, optionally test:

```text
configs/minimax_h3_t2av_dmd_a100_2x8.yaml
```

It uses the same checkpoint-boundary policy. Keep global batch16 and 16 independent rank samples; HSDP changes parameter placement only.

## 11. Final artifacts

Write/update:

```text
docs/experiments/H3_DMD_A100_CONTROLLED_FULL5.md
artifacts/h3_dmd_a100_controlled_full5.json
```

Include:

```text
exact source HEAD
activation policy and checkpoint segment
31-forward/6-backward matched census
6-grad-forward/300-boundary-copy census
capacity status
OOM location if any
peak allocated/reserved/free HBM
host/cgroup peak
one-cycle training wall
FULL5 run1/run2/run3 + median/CV
Student/Fake phase times
boundary D2H/H2D bytes
correctness/state status
```

If the new shared-backbone boundary policy cannot fit, preserve the evidence and continue with residency attribution; do not silently substitute a smaller workload.
