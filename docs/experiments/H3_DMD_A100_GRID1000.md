# MiniMax-H3 A100 Grid-1000 AdaLN Cache Experiment

Run IDs: `iteration_310_grid1000_world16_20260820` (diagnostic) and
`iteration_311_grid1000_lowoverhead_20260820` (low-overhead one-cycle run).

## Contract

- 2 nodes x 8 A100-SXM4-40GB, world16 FSDP2 FULL_SHARD
- MiniMax-H3, 768x1344x124, BF16, B1/rank
- fixed `end_step_idx=3`, `_flash_3_hub`
- one outer DMD cycle = Student update 1 + Fake updates 5
- application census: Student/Fake/Teacher = 24/6/1; backward = 1/5
- native per-block checkpointing (`segment_size=1`) with `checkpoint_boundary_cpu`
- no segment=8, no generic threshold activation offload, no parameter offload/model swap
- Grid-1000 sigma is an approximation branch: base sigma is snapped to a frozen 1000-point grid.

Source HEAD: `cf141e6edb2bc1bfdf0f09d5e4c88842a5cc049f`
LightX2V HEAD: `d034a6b0ecaa31aa07c81aeb7bbe69b225f1d7be`

## Table gate

The table was built with the per-entry/two-timestep-pair builder in
`scripts/build_grid1000_adaln_table.py` (builder SHA256
`1901386c9ecfb7f88eac217f2875ce55f804e7bbeb123e05440db303f5227481`).
Per-entry construction is required because a single giant BF16 GEMM did not
produce bitwise-equal values to the runtime two-row request shape.

- binary bytes: `19,431,014,400` (18.10 GiB logical mmap)
- dropped AdaLN parameter numel: `13,010,457,600`
- table manifest SHA256: `dd2625e0013509e31b984ee37be752c5d397da7e23aa1c945f66150af95cc1f3`
- validator: **PASS**, 48 checks, zero failures
- runtime GPU cache observed: 73.83 MiB; no full 18.1 GiB GPU copy

## World16 results

The diagnostic run enabled memory-attribution logging; its wall envelope is
not a formal timing result. The low-overhead run disabled that instrumentation.

| Run | Start (UTC) | End (UTC) | One-cycle wall | Peak allocated | Peak reserved | OOM delta | Gates |
|---|---:|---:|---:|---:|---:|---:|---|
| Grid diagnostic | 06:33:15.010 | 06:48:20.903 | 905.893 s | 17.56 GiB | 20.12 GiB | 0 | PASS |
| Grid low-overhead | 06:58:50.253 | 07:14:02.018 | **911.765 s** | 17.56 GiB | 20.12 GiB | 0 | PASS |

Both runs report `world=16`; rank-0 validation is emitted only after the
world-wide gathers complete. No rank reported a contract exception.

Low-overhead gate receipts:

- forward: Student 24, Fake 6, Teacher 1
- grad-forward: Student 1, Fake 5, Teacher 0
- backward: Student 1, Fake 5
- checkpoint-boundary: 6 grad transformer forwards, 300 checkpoint calls, 300 CPU copies
- Grid replay: 300 checkpoint wrappers, 600 scoped executions, 600 cache hits, 0 misses, 0 missing keys
- dropped AdaLN parameters: 13,010,457,600; registered `.adaln_proj.` parameters: 0
- final HBM after cleanup: 2.92 GiB allocated, 20.12 GiB reserved; peak allocated 17.56 GiB

Node cleanup after the run: all 16 GPUs reported 0 MiB used and no training
workers remained. cgroup `oom` remained 0 on both nodes (node0 historical
`oom_kill=4` and node1 `oom_kill=0` were unchanged).

## Comparison

The exact replay branch low-overhead one-cycle reference was 912.718721 s and
19.01 GiB peak allocated. The Grid low-overhead run was 911.765 s, a raw
single-run difference of -0.954 s (-0.10%). This is **not a formal speedup
claim**: Grid changes the sigma distribution and is approximate, and only one
run per variant is available.

DMD-System's exact successful reference remains 917.993973 s / 27.94 GiB
(native) and 914.102536 s / 28.10 GiB (phase-cluster); these are useful context
but not matched numerical comparisons to Grid-1000.

## Status and limits

`CAPACITY_PASS` and `CONTRACT_PASS` are established for one outer cycle. A
three-run timing study and continuous-sigma numerical-quality study are **NOT
RUN**. The 18.1 GiB table binary is kept on the shared asset path and is not
committed to GitHub; only its manifest/hash and validation receipt are tracked.
