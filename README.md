# H3-DMD-for-A100

A pinned LightX2V overlay for running MiniMax-H3 DMD on **two nodes of 8×A100-40GB** while preserving a fail-closed matched-compute contract against the DMD-System MiniMax baseline.

The upstream implementation is intentionally not copied into this repository. `scripts/bootstrap_lightx2v.sh` checks out the exact LightX2V commit in [`UPSTREAM_COMMIT`](UPSTREAM_COMMIT), and this package registers an optimized H3 model/trainer alongside the upstream code.

> **Hardware-validation status:** source/unit checks are complete; the controlled world16 A100 run is still required. Do not report speedup unless the per-cycle matched-compute census passes on all 16 ranks.

## Controlled benchmark contract

The primary two-node config is:

```text
configs/minimax_h3_t2av_dmd_a100_world16.yaml
```

It matches the DMD-System reference on the training workload:

```text
2 nodes x 8 A100-40GB
world16 FSDP2
768 x 1344 x 124 frames
B1 per global rank / global batch16
Student LoRA rank128
Fake LoRA rank128
4 Student rollout evaluations per back-simulation
Student update 1 + Fake updates 5
continuous renoise sigma
BF16 model compute
_flash_3_hub attention backend
```

Every outer cycle fails closed unless each rank records exactly:

```text
Student DiT forwards          24
Fake DiT forwards              6
Teacher DiT forwards           1
Student grad forwards          1
Fake grad forwards             5
Student backward graphs        1
Fake backward graphs           5
```

The six stage samples (`Student + Fake0..4`) are gathered after the cycle and must contain **96 distinct rank-qualified identities** across world16. Activation-checkpoint recomputation is not counted as a new application forward.

## Systems changes under test

### One physical backbone for three logical DMD roles

```text
                         one frozen H3 base
                    ┌──────────┼──────────┐
                    │          │          │
              student LoRA  fake LoRA   adapters off
                    │          │          │
                 Student      Fake       Teacher
```

Student and Fake keep disjoint optimizers/schedulers. Teacher is the same frozen base with adapters disabled. The backward path re-enters the correct role scope so activation-checkpoint recomputation cannot inherit the wrong adapter.

### Exact AdaLN precomputation

H3's large block-level AdaLN projections depend on timestep/modality rather than prompt/latent content. The overlay extracts them into separately shardable units:

- four fixed rollout timestep tables persist across training;
- one random DMD score table is computed once and shared by Fake and Teacher for that score query;
- the continuous random-sigma objective is unchanged;
- no sigma discretization is used in the primary matched run.

### Critic-only phase clustering

The optimized loop preserves five distinct Fake optimizer commits while changing only scheduling:

```text
Student update
→ Student no-grad rollout x5
→ Fake update 1
→ Fake update 2
→ Fake update 3
→ Fake update 4
→ Fake update 5
```

### Optional HSDP placement ablation

`configs/minimax_h3_t2av_dmd_a100_2x8.yaml` uses a 2×8 HSDP mesh to keep parameter all-gathers node-local. It is **not** the primary denominator. HSDP changes placement only: all 16 global ranks still receive independent B1 samples and independent rank-seeded RNG streams, so global batch remains 16.

## Source identity

Controlled runs pin:

```text
LightX2V: d034a6b0ecaa31aa07c81aeb7bbe69b225f1d7be
Diffusers: 0.40.0.dev0
Diffusers source: 9284607295a09f759aadd65ed08f48b35feea6d9
attention backend: _flash_3_hub
```

`scripts/preflight.py` fails before model construction if these identities or the visible A100 topology do not match.

## Quick start

```bash
git clone https://github.com/sliortega295-ops/H3-DMD-for-A100.git
cd H3-DMD-for-A100
git checkout agent/h3-a100-shared-backbone
bash scripts/bootstrap_lightx2v.sh
source .h3-a100-env

export MINIMAX_H3_MODEL_PATH=/shared/models/MiniMax-H3
export H3_PROMPT_CACHE=/shared/datasets/minimax_h3_prompt_cache
export H3_OUTPUT_DIR=/shared/outputs/h3_dmd_a100
export H3_ATTN_BACKEND=_flash_3_hub
export H3_BENCHMARK_SEED=42
```

Run preflight on both nodes, then a one-node smoke test:

```bash
python scripts/preflight.py --lightx2v-root "$LIGHTX2V_ROOT"
bash scripts/smoke_1node.sh
```

For the primary two-node controlled run, `scripts/launch_2node.sh` defaults to the world16 FSDP2 config. Set `NODE_RANK` and `MASTER_ADDR` on each node as documented in `docs/RUNBOOK_2NODE_A100.md`.

To run the HSDP placement ablation explicitly:

```bash
export CONFIG=$PWD/configs/minimax_h3_t2av_dmd_a100_2x8.yaml
```

## Tests

```bash
bash scripts/run_unit_tests.sh
```

The tests cover exact AdaLN cache reuse/eviction, frozen-weight validation, 2×8 topology mapping, matched application-level forward/backward census, global 96-sample uniqueness, pinned-upstream contracts, config invariants, and shell syntax.

## Current limitations

- The current CPU-first loader may still instantiate a full CPU H3 module per local rank before FSDP sharding; host-memory-safe direct DTensor loading remains follow-up work if cold construction OOMs.
- The primary exact run keeps continuous renoise sigma, so AdaLN score outputs are cached per sampled sigma rather than replaced by a finite lookup grid.
- Checkpoint/resume remains same-topology for this first implementation.
- Full-resolution A100 timing is not claimable until the matched census, state/optimizer checks, and real 16-GPU run pass.
