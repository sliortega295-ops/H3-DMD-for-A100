# Two-node A100 runbook

## Target topology

- 2 nodes
- 8×A100-40GB or A800-40GB per node
- contiguous torchrun rank assignment by node
- NVLink/NVSwitch inside each node
- working NCCL transport between nodes
- shared filesystem for model, prompt cache, output checkpoints
- large host RAM; the current CPU-first loader may approach the node's memory limit during initialization

## 1. Prepare an identical checkout on both nodes

```bash
git clone https://github.com/sliortega295-ops/H3-DMD-for-A100.git
cd H3-DMD-for-A100
bash scripts/bootstrap_lightx2v.sh
source .h3-a100-env
```

Both nodes must report the same pinned LightX2V commit:

```bash
git -C "$LIGHTX2V_ROOT" rev-parse HEAD
cat UPSTREAM_COMMIT
```

## 2. Environment

Use the CUDA/PyTorch/Diffusers environment required by the pinned LightX2V H3 trainer. In particular, the Diffusers build must contain `MiniMaxH3Transformer3DModel` with PEFT adapter and attention-backend APIs.

Export identical shared paths on both nodes:

```bash
export MINIMAX_H3_MODEL_PATH=/shared/models/MiniMax-H3
export H3_PROMPT_CACHE=/shared/datasets/minimax_h3_prompt_cache
export H3_OUTPUT_DIR=/shared/outputs/h3_dmd_a100
export H3_ATTN_BACKEND=flash
```

Optional NCCL settings depend on the cluster:

```bash
export NCCL_SOCKET_IFNAME=<network-interface>
export NCCL_IB_HCA=<ib-devices>
export NCCL_DEBUG=WARN
```

Do not copy interface names from another cluster blindly.

## 3. Preflight on both nodes

```bash
python scripts/preflight.py --lightx2v-root "$LIGHTX2V_ROOT"
```

Hard failures include:

- wrong LightX2V commit;
- fewer or more than 8 visible GPUs;
- missing H3 Diffusers APIs;
- wrong model layout;
- missing prompt cache.

Host-memory and unexpected-GPU messages are warnings so the report can still be collected.

## 4. One-node smoke gate

Run this before reserving two nodes for a long job:

```bash
bash scripts/smoke_1node.sh 2>&1 | tee smoke.log
```

Required pass conditions:

1. full H3 never attempts a pre-FSDP `.to(cuda)`;
2. setup finishes on all 8 ranks without host/GPU OOM;
3. one Student step is finite;
4. five Fake optimizer steps complete;
5. AdaLN stats show persistent cache entries and increasing hits;
6. no rank divergence, collective timeout, or adapter-missing error.

The smoke workload is 960×544, 40 frames, one training iteration. It validates control flow, not final memory at 1344×768×124.

## 5. Optional Nsight Systems trace

```bash
bash scripts/profile_nsys.sh
```

Relevant NVTX ranges:

- `h3/student_step`
- `h3/critic_phase`
- `h3/critic_prepare_5xG`
- `h3/critic_update_F`

Check that the five G rollouts form one contiguous region and that cross-node NCCL traffic is dominated by replica-gradient communication rather than every block's parameter all-gather.

## 6. Two-node launch

Choose a node-0 address reachable from node 1.

Node 0:

```bash
NODE_RANK=0 MASTER_ADDR=<node0-ip> MASTER_PORT=29500 \
  bash scripts/launch_2node.sh 2>&1 | tee node0.log
```

Node 1:

```bash
NODE_RANK=1 MASTER_ADDR=<node0-ip> MASTER_PORT=29500 \
  bash scripts/launch_2node.sh 2>&1 | tee node1.log
```

The launcher creates a 2×8 HSDP mesh. Do not change `NPROC_PER_NODE=8` without also changing `distributed.hybrid_shard.shard_size`.

## 7. First full-resolution gate

Before 1000 iterations, run a short full-resolution job:

```bash
export H3_MAX_ITERS=3
export H3_SAVE_EVERY=1
# launch both nodes as above
```

Inspect:

- peak GPU memory from `[h3-a100][memory]` logs;
- node host-memory peak;
- finite Student/Fake losses;
- checkpoint creation and same-topology resume;
- AdaLN cache memory and hit counts;
- iteration wall time and five critic updates.

Then restart from the last checkpoint with a larger `H3_MAX_ITERS`.

## 8. Troubleshooting

### OOM before FSDP setup finishes

This is a host-memory loading peak, not a CUDA activation problem. Confirm the custom model name is selected and logs say the full H3 checkpoint stayed on CPU. Close competing jobs and inspect per-rank RSS. The next engineering step is direct sharded/meta loading; reducing resolution will not fix initialization RAM.

### CUDA OOM during the first forward

Start with the smoke config. Verify HSDP shard size is 8 and that all 8 GPUs are visible. Inspect whether FlashAttention fell back to native SDPA. Reduce frame count/canvas only for diagnosis; the paper configuration remains 1344×768×124.

### Different samples/noise inside an HSDP shard group

All ranks 0–7 must share data-parallel rank 0; ranks 8–15 must share data-parallel rank 1. The custom entrypoint is required—running upstream `lightx2v_train/train.py` bypasses this mapping and is incorrect for the 2-D mesh.

### Hang during AdaLN precompute

Each AdaLN projection is a separate FSDP unit and all ranks in the shard group must call it in the same order. A rank-specific exception before precompute usually appears as a later NCCL timeout; inspect the earliest rank log.

### Resume failure

The v1 checkpoint is same-topology only. Resume with the same 16 ranks, 2×8 mesh, upstream commit, model checkpoint, and LoRA configuration.
