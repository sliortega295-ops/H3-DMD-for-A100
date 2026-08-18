# H3-DMD-for-A100

An optimization overlay for running MiniMax-H3 LoRA-DMD on **two 8×A100-40GB nodes**. It pins a known LightX2V commit and replaces the three-physical-model DMD layout with one shared H3 backbone plus two named LoRA adapters.

## What changes

The upstream H3 DMD path constructs three independent 33B transformers:

```text
Student = H3 base + student LoRA
Fake    = H3 base + fake LoRA
Teacher = H3 base
```

This overlay executes the same logical roles through one physical transformer:

```text
                         one frozen H3 base
                    /           |           \
        student adapter    fake adapter    adapters off
              G                 F               T
```

The first implementation includes seven system optimizations:

1. **Shared backbone.** Student and Fake use named PEFT adapters; Teacher disables adapters. Only one 33B base is loaded and FSDP/HSDP-sharded.
2. **Exact AdaLN precomputation.** The 50 large `adaln_proj` modules are extracted into a separately shardable bank. Fixed four-step rollout modulations are cached permanently; a random DMD score modulation is computed once and reused by Fake and Teacher. Cache tensors are detached only after runtime validation confirms that AdaLN and the timestep MLP are frozen.
3. **FSDP2-safe Student replay.** The conservative default scores a no-grad rollout with Fake/Teacher, then repeats only the sampled final Student evaluation immediately before backward. This avoids holding an FSDP2 graph across additional role forwards. A `retain` fast path removes the extra final NFE after the included capability probe passes on the target PyTorch build.
4. **Critic-only reordering.** For `fake_update_ratio=5`, all five frozen-G rollouts are generated first, followed by five sequential Fake optimizer steps. This preserves optimizer-step semantics while improving role locality.
5. **Two-node HSDP.** Parameters are sharded across the eight GPUs inside each node and replicated across the two nodes. Weight all-gathers stay in the NVLink domain instead of crossing nodes for every block, while all 16 GPUs remain independent data-parallel workers.
6. **A100-safe kernels.** The default attention backend is installed FlashAttention-2 (`flash`), which targets Ampere/A100. The Hopper-oriented upstream `_flash_3_hub` default is not used.
7. **A100-safe initialization.** The 66 GB BF16 H3 checkpoint remains on CPU until FSDP2 converts parameters into local DTensor shards. Per-node loading is serialized by local rank to reduce storage contention and transient host-memory spikes; optional pinned activation offload is enabled in the first-run config.

The repository is an overlay rather than a source fork. `scripts/bootstrap_lightx2v.sh` checks out the pinned upstream commit; `PYTHONPATH` loads the custom model and trainer without modifying that checkout.

## Quick start

```bash
./scripts/bootstrap_lightx2v.sh

export MINIMAX_H3_MODEL_PATH=/shared/models/MiniMax-H3
export H3_PROMPT_CACHE_PATH=/shared/data/minimax_h3_prompt_cache
export H3_OUTPUT_DIR=/shared/checkpoints/h3_a100_shared
```

On node 0:

```bash
MASTER_ADDR=10.0.0.10 NODE_RANK=0 ./scripts/launch_2node.sh
```

On node 1:

```bash
MASTER_ADDR=10.0.0.10 NODE_RANK=1 ./scripts/launch_2node.sh
```

Before loading H3, run the environment preflight and the tiny FSDP2 shared-backbone probe on the exact cluster environment:

```bash
./scripts/preflight.sh
PROBE_GPUS=2 ./scripts/probe_fsdp2.sh
```

The production config defaults to `H3_STUDENT_GRAPH_MODE=replay`, which does not depend on the retain-graph probe. After the probe passes, benchmark the faster path with `H3_STUDENT_GRAPH_MODE=retain`.

Run the 8-GPU, two-iteration smoke test before the two-node job. The script defaults to a valid diagnostic shape of 256×448 and 22 frames:

```bash
H3_MAX_ITERS=2 ./scripts/smoke_1node.sh
```

See [the two-node runbook](docs/RUNBOOK_2NODE_A100.md) for the required validation sequence, [the design document](docs/DESIGN.md) for correctness boundaries, and [the optimization backlog](docs/OPTIMIZATION_BACKLOG.md) for the next evidence-driven changes.

## Checkpoint format

Each checkpoint contains:

```text
checkpoint-000000100/
├── student/pytorch_lora_weights.safetensors
├── fake/pytorch_lora_weights.safetensors
├── rank-0000.pt ... rank-0015.pt
├── trainer_state.pt
└── _SUCCESS
```

LoRA weights are portable. Optimizer state is rank-local and currently requires the same world size and HSDP topology when resuming.

## Current validation status

Completed in this change:

- exact-cache and timestep-signature unit tests;
- replay and critic-reorder equivalence unit tests;
- two-node topology unit tests;
- a two-rank CPU/Gloo FSDP2 shared-backbone state-machine probe on PyTorch 2.10;
- Python bytecode compilation;
- shell syntax validation for all launch/probe/preflight scripts;
- config and launch-script construction against LightX2V commit `e4ac7ef0122b79ea75b4af429a34f40456b741d4`.

Not available in this execution environment:

- CUDA/FSDP execution;
- A100 memory traces;
- numerical comparison against upstream LightX2V;
- a full two-node training iteration.

The first cluster run should therefore be treated as a controlled bring-up, not as a finished performance result.
