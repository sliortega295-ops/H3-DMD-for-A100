# Two-node A100 runbook

## Target topology and comparison rule

- 2 nodes
- 8×A100-40GB or A800-40GB per node
- contiguous torchrun rank assignment by node
- working NCCL between nodes
- shared model/prompt-cache/output storage

The **primary controlled run** uses `configs/minimax_h3_t2av_dmd_a100_world16.yaml`: world16 FSDP2, B1 per global rank, global batch16. The optional `..._2x8.yaml` HSDP config is a placement ablation only and must preserve the same 16 independent samples/RNG streams.

Every formal outer cycle must pass the matched-compute census:

```text
Student forward 24 / grad-forward 1 / backward 1
Fake forward     6 / grad-forward 5 / backward 5
Teacher forward  1 / grad-forward 0
96 unique sample identities across 16 ranks x 6 stages
fixed end_step_idx = 3
```

If any count/sample identity differs, the run is invalid regardless of wall time.

## 1. Prepare identical checkout on both nodes

```bash
git clone https://github.com/sliortega295-ops/H3-DMD-for-A100.git
cd H3-DMD-for-A100
git checkout agent/h3-a100-shared-backbone
bash scripts/bootstrap_lightx2v.sh
source .h3-a100-env
```

The pinned LightX2V revision must be:

```text
d034a6b0ecaa31aa07c81aeb7bbe69b225f1d7be
```

## 2. Environment

```bash
export MINIMAX_H3_MODEL_PATH=/shared/models/MiniMax-H3
export H3_PROMPT_CACHE=/shared/datasets/minimax_h3_prompt_cache
export H3_OUTPUT_DIR=/shared/outputs/h3_dmd_a100
export H3_ATTN_BACKEND=_flash_3_hub
export H3_BENCHMARK_SEED=42
```

Controlled runs require Diffusers `0.40.0.dev0` at source revision `9284607295a09f759aadd65ed08f48b35feea6d9`. `scripts/preflight.py` fails closed on source/backend mismatch.

Cluster-specific NCCL variables may be set as needed, but do not copy interface/HCA names from another cluster blindly.

## 3. Preflight on both nodes

```bash
python scripts/preflight.py --lightx2v-root "$LIGHTX2V_ROOT"
```

Failures before model construction must be classified separately from CUDA/host OOM.

## 4. One-node smoke

```bash
bash scripts/smoke_1node.sh 2>&1 | tee smoke.log
```

Required:

1. setup finishes on all 8 ranks;
2. one Student + five Fake updates finish;
3. matched census is `24/6/1` forward and `1/5` backward per rank;
4. 48 unique sample identities are observed for the 8-rank smoke;
5. AdaLN cache hits increase;
6. no adapter-state or collective divergence.

## 5. Primary two-node capacity gate

Use the default world16 config and exactly one outer cycle first:

```bash
export H3_MAX_ITERS=1
export H3_SAVE_EVERY=0
```

Node 0:

```bash
NODE_RANK=0 MASTER_ADDR=<node0-ip> MASTER_PORT=29500 \
  bash scripts/launch_2node.sh 2>&1 | tee node0.capacity.log
```

Node 1:

```bash
NODE_RANK=1 MASTER_ADDR=<node0-ip> MASTER_PORT=29500 \
  bash scripts/launch_2node.sh 2>&1 | tee node1.capacity.log
```

Do not reduce resolution, frames, LoRA rank, Fake update count, or end-step to make the gate fit.

## 6. Five-cycle timing

Only after the one-cycle capacity/correctness gate passes:

```bash
export H3_MAX_ITERS=5
export H3_SAVE_EVERY=0
```

Formal wall-clock measurement must exclude cold loading/FSDP construction. Synchronize CUDA/world16 immediately before the first Student update and immediately after the fifth cycle's fifth Fake optimizer update. Preserve per-cycle phase timings separately.

The matched census remains enabled during timing; its global sample check occurs after each cycle and is correctness evidence, not a license to change the compute graph.

## 7. HSDP ablation

After the primary world16 result exists:

```bash
export CONFIG=$PWD/configs/minimax_h3_t2av_dmd_a100_2x8.yaml
```

Run the same one-cycle capacity gate and five-cycle timing. HSDP is comparable only if the same matched census and world16 sample contract pass.

## 8. OOM follow-up order

If the primary world16 shared-backbone exact run CUDA-OOMs, preserve the exact failure location/allocator counters and test memory mechanisms one at a time. Do not mix multiple changes in the first retry. Recommended order is selective activation checkpoint/offload policy first, then context/sequence parallelism if activation remains dominant. Sigma discretization/AdaLN weight dropping is a separate algorithmic ablation and not part of the exact matched baseline.

If cold construction hits host/cgroup OOM, fix initialization (meta/direct sharded loading or shared immutable backing) without changing the timed workload.

## 9. Nsight

Capture Nsight only after an unprofiled stable result exists. Profiled wall time is diagnostic, not the formal denominator. Keep CUDA compute, NCCL, H2D and D2H interval unions separate because they may overlap.
