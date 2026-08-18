# H3-DMD-for-A100

A pinned overlay for running MiniMax-H3 DMD on **two nodes of 8×A100-40GB** with LightX2V.

The upstream implementation is intentionally not copied into this repository. `scripts/bootstrap_lightx2v.sh` checks out the exact LightX2V commit in [`UPSTREAM_COMMIT`](UPSTREAM_COMMIT), and this package registers an optimized H3 model/trainer alongside the upstream code.

> **Validation status:** CPU unit tests and source-contract checks pass. The code has not yet been executed on the target A100 cluster, so the first one-node smoke test is still a required gate before a full two-node run.

## What changes

### One physical backbone for three DMD roles

Upstream H3-DMD constructs three independent 33B transformers:

```text
Student = H3 base + student LoRA
Fake    = H3 base + fake LoRA
Teacher = H3 base, frozen
```

This overlay constructs one transformer and two named PEFT adapters:

```text
                         one frozen H3 base
                    ┌──────────┼──────────┐
                    │          │          │
              student LoRA  fake LoRA   adapters off
                    │          │          │
                 Student      Fake       Teacher
```

The model is FSDP/HSDP-wrapped once. Student and Fake keep separate parameter lists, optimizers, learning-rate schedulers, and checkpoint files.

### Exact AdaLN precomputation

H3's 50 large block-level AdaLN projections depend on the timestep table, not the prompt, seed, or latent trajectory. They are moved into a separately shardable bank.

- The four fixed rollout timestep tables are computed once and kept for all iterations.
- A random DMD score timestep is computed once and reused by Fake and Teacher in that update.
- The cache scope is restored during gradient-checkpoint recomputation, so backward sees the exact same modulation tensors.
- The optimization is rejected if AdaLN or the timestep MLP becomes trainable.

### Critic-only rollout reordering

The default LightX2V loop is:

```text
G rollout → F update → G rollout → F update → ... ×5
```

During these five Fake updates, G is frozen. The optimized loop is:

```text
G rollout ×5 → F update ×5
```

The five Fake optimizer steps remain distinct and sequential. This improves role locality without turning five updates into one larger-batch update.

### Two-node HSDP topology

The recommended 16-GPU mesh is:

```text
replicate dimension: 2 nodes
shard dimension:     8 GPUs inside each node
```

Large parameter all-gathers stay inside the node/NVLink domain; gradients are replicated across the two nodes. Inputs and RNG draws are synchronized inside each 8-GPU shard group, while the two replicas consume different samples.

### CPU-first FSDP initialization

The upstream wrapper moves the complete BF16 transformer to one GPU before FSDP. This overlay leaves the full checkpoint on CPU and lets FSDP2/HSDP move only local DTensor shards to CUDA. After sharding, Python and glibc caches are trimmed to release transient host pages.

## Repository layout

```text
h3_a100/
  model.py          shared backbone, named adapters, AdaLN bank
  trainer.py        DMD loop, rollout reordering, checkpointing
  adaln_cache.py    exact persistent/dynamic modulation cache
  distributed.py    2×8 HSDP topology and shard-group RNG sync
  entrypoint.py     LightX2V-compatible training entrypoint
configs/
  ..._smoke.yaml    one-node functional gate
  ..._2x8.yaml      two-node training configuration
scripts/
  bootstrap_lightx2v.sh
  preflight.py
  smoke_1node.sh
  launch_2node.sh
  profile_nsys.sh
```

## Quick start

### 1. Bootstrap the pinned upstream checkout

```bash
git clone https://github.com/sliortega295-ops/H3-DMD-for-A100.git
cd H3-DMD-for-A100
bash scripts/bootstrap_lightx2v.sh
source .h3-a100-env
```

Install the H3 training dependencies expected by the pinned LightX2V revision in the existing CUDA environment. The preflight script checks the important APIs and packages.

### 2. Export shared paths

```bash
export MINIMAX_H3_MODEL_PATH=/shared/models/MiniMax-H3
export H3_PROMPT_CACHE=/shared/datasets/minimax_h3_prompt_cache
export H3_OUTPUT_DIR=/shared/outputs/h3_dmd_a100
export H3_ATTN_BACKEND=flash
```

`MINIMAX_H3_MODEL_PATH` must be the converted Diffusers layout containing `transformer/config.json` with class `MiniMaxH3Transformer3DModel`.

### 3. Run preflight on each node

```bash
python scripts/preflight.py --lightx2v-root "$LIGHTX2V_ROOT"
```

### 4. One-node functional smoke test

```bash
bash scripts/smoke_1node.sh
```

The smoke test runs one Student update and five sequential Fake updates on 8 GPUs at 960×544 and 40 frames.

### 5. Two-node run

On node 0:

```bash
NODE_RANK=0 MASTER_ADDR=<node0-ip> MASTER_PORT=29500 \
  bash scripts/launch_2node.sh
```

On node 1:

```bash
NODE_RANK=1 MASTER_ADDR=<node0-ip> MASTER_PORT=29500 \
  bash scripts/launch_2node.sh
```

See [the two-node runbook](docs/RUNBOOK_2NODE_A100.md) before starting a long run.

## Tests

```bash
bash scripts/run_unit_tests.sh
```

The tests cover exact AdaLN cache reuse/eviction, frozen-weight validation, 2×8 topology mapping, pinned-upstream contracts, config invariants, and shell syntax.

## Current limitations

- The checkpoint format is **same-topology resume only**. Optimizer states are rank-local; resume with the same world size and mesh.
- CDM, SenseFlow IDA, diversity loss, and real-data Fake branches are deliberately disabled in the first shared-backbone version.
- CPU-first loading still creates one full CPU H3 module per rank before sharding. Sharing or streaming checkpoint pages across local ranks is a later optimization; nodes should have large host memory.
- PEFT adapter switching inside one FSDP2 module and 2-D HSDP need target-hardware validation. The repository therefore opens as a draft implementation, not as a claim of completed A100 reproduction.

## Design and validation documents

- [System design and semantic invariants](docs/DESIGN.md)
- [Two-node A100 runbook](docs/RUNBOOK_2NODE_A100.md)
- [Ablation and validation plan](docs/EXPERIMENT_PLAN.md)
