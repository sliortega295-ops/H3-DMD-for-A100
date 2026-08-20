# Exact AdaLN replay: one-cycle result

Run: `iteration_309_exact_adaln_replay_timing_20260820`

This is the controlled **one outer DMD cycle**, defined as **Student update 1 + Fake updates 5**. It is not a five-cycle run.

## Workload

- 2 nodes × 8 A100-SXM4-40GB, world16
- MiniMax-H3 768×1344×124, BF16, B1/rank
- FSDP2 FULL_SHARD, sequence parallel 1
- shared physical backbone; Student/Fake LoRA rank128; Teacher adapters disabled
- fixed `end_step_idx=3`, 4-step rollout, continuous sigma
- native per-block checkpointing (`segment=1`)
- `checkpoint_boundary_cpu`, pinned CPU boundary staging
- exact AdaLN checkpoint replay; no segment=8 and no generic threshold offload

## Measured result

The low-overhead one-cycle run measured from `train start` to `train finished`:

- **one-cycle wall: 912.718721 s**
- Student forwards: 24; Fake forwards: 6; Teacher forwards: 1
- Student backward: 1; Fake backward: 5
- checkpoint boundary calls: 300; CPU copies: 300
- exact replay scopes: 600; cache hits: 600; cache misses: 0
- observed rank-0 peak allocated: 19.01 GiB
- no CUDA or cgroup OOM; 16/16 workers completed

The detailed per-node timing logs are kept in `raw/node0_timing.log` and `raw/node1_timing.log`.

## Comparison references

These are historical results from the DMD-System MiniMax-H3 reference and are not re-run in this artifact:

| Implementation | One-cycle wall | Peak allocated | Status |
|---|---:|---:|---|
| DMD-System Native | 917.993973 s | 27.94 GiB | PASS |
| DMD-System phase-cluster | 914.102536 s | 28.10 GiB | PASS |
| H3 shared-backbone Exact replay | **912.718721 s** | **19.01 GiB (rank-0 observed)** | PASS |

The comparison is descriptive, not a new formal speedup claim: timer implementations and schedule organization differ. The previous `H3_MAX_ITERS=5` process was stopped after its first completed cycle because Full5 here means one Student + five Fake updates; its logs are preserved under the research-artifact directory and are not used as a result.
