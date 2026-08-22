# H3-DMD-for-A100

A pinned LightX2V overlay for running MiniMax-H3 DMD on **2 nodes × 8 A100-40GB** with one shared physical H3 backbone while preserving a fail-closed matched-compute contract against the DMD-System MiniMax workload.

The upstream implementation is intentionally not copied into this repository. `scripts/bootstrap_lightx2v.sh` checks out the exact LightX2V commit recorded in [`UPSTREAM_COMMIT`](UPSTREAM_COMMIT), and this package registers the A100-oriented H3 model/trainer alongside that pinned upstream source.

## Current validated status

The primary **Exact Replay** path has completed the controlled world16 one-cycle workload:

```text
2 nodes × 8 A100-SXM4-40GB
world16 FSDP2 FULL_SHARD
768 × 1344 × 124 frames
BF16, B1/rank
Student LoRA rank128
Fake LoRA rank128
Teacher = shared frozen base, adapters disabled
4 rollout evaluations, fixed end_step_idx=3
continuous renoise sigma
Student update 1 + Fake updates 5
_flash_3_hub
```

Measured low-overhead one-cycle result:

```text
wall                         912.718721 s
peak allocated               19.01 GiB/rank (rank-0 observed)
Student/Fake/Teacher fwd      24 / 6 / 1
Student/Fake backward          1 / 5
checkpoint calls             300
boundary CPU copies          300
Exact replay scopes          600
AdaLN cache hits/misses      600 / 0
CUDA/cgroup OOM                0
workers completed             16 / 16
```

See [`docs/experiments/H3_DMD_A100_EXACT_REPLAY.md`](docs/experiments/H3_DMD_A100_EXACT_REPLAY.md) and [`artifacts/h3_dmd_a100_exact_replay.json`](artifacts/h3_dmd_a100_exact_replay.json).

Historical DMD-System references are 917.993973 s / 27.94 GiB for Native and 914.102536 s / 28.10 GiB for phase-cluster. Those comparisons are **descriptive only**, not a formal speedup claim, because the timer implementations and schedule organization are not identical.

## Primary implementation

### One physical backbone for three logical roles

```text
                         one frozen H3 base
                    ┌──────────┼──────────┐
                    │          │          │
              Student LoRA  Fake LoRA   adapters off
                    │          │          │
                 Student      Fake       Teacher
```

Student and Fake keep disjoint optimizer/scheduler state. Teacher reuses the same frozen backbone with adapters disabled.

### Exact AdaLN precomputation and replay

MiniMax-H3 has very large block-level AdaLN projections whose outputs depend only on timestep/modality, not on prompt tokens, latent content, or seed. The exact path:

- precomputes the four fixed rollout timestep tables and reuses them;
- computes dynamic DMD-score AdaLN once per exact sampled sigma and shares that table between Fake and Teacher;
- preserves the original continuous-sigma DMD objective;
- captures the exact AdaLN cache key inside native non-reentrant checkpoint functions so backward replay cannot fall back to rematerializing the giant AdaLN projections;
- fails closed on missing/incomplete replay keys or any AdaLN replay cache miss.

This fixes the Iteration 308 failure mode where backward replay accumulated roughly one full ~520 MB AdaLN projection per block and OOMed above 37 GiB.

### Native per-block checkpointing + boundary CPU staging

The validated exact configuration uses:

```text
activation checkpoint segment = 1
activation policy             = checkpoint_boundary_cpu
50 native checkpoint calls / grad-enabled H3 forward
```

Only each checkpoint input boundary is staged to pinned CPU. Recomputed FFN/attention/LoRA intermediates are not recursively offloaded.

A matched outer cycle contains one Student grad-enabled H3 forward and five Fake grad-enabled H3 forwards, so the runtime fails closed unless each rank observes exactly 300 checkpoint calls and 300 boundary CPU copies.

## Controlled benchmark contract

Primary config:

```text
configs/minimax_h3_t2av_dmd_a100_world16.yaml
```

Every outer cycle must record exactly:

```text
Student DiT forwards          24
Fake DiT forwards              6
Teacher DiT forwards           1
Student grad forwards          1
Fake grad forwards             5
Teacher grad forwards          0
Student backward graphs        1
Fake backward graphs           5
```

The six stage samples (`Student + Fake0..4`) are gathered after the cycle and must contain **96 distinct rank-qualified identities** across world16. Activation-checkpoint recomputation is not counted as a new application forward.

## Source identity

Controlled runs pin:

```text
LightX2V: d034a6b0ecaa31aa07c81aeb7bbe69b225f1d7be
Diffusers source: 9284607295a09f759aadd65ed08f48b35feea6d9
attention backend: _flash_3_hub
benchmark seed: 20260817
```

`scripts/preflight.py` validates the pinned software/source identity and visible A100 topology before the formal workload.

## Quick start

```bash
git clone https://github.com/sliortega295-ops/H3-DMD-for-A100.git
cd H3-DMD-for-A100
bash scripts/bootstrap_lightx2v.sh
source .h3-a100-env

export MINIMAX_H3_MODEL_PATH=/shared/models/MiniMax-H3
export H3_PROMPT_CACHE=/shared/datasets/minimax_h3_prompt_cache
export H3_OUTPUT_DIR=/shared/outputs/h3_dmd_a100
export H3_ATTN_BACKEND=_flash_3_hub
export H3_BENCHMARK_SEED=20260817
```

Run validation first:

```bash
bash scripts/run_unit_tests.sh
python scripts/preflight.py --lightx2v-root "$LIGHTX2V_ROOT"
```

For a one-node diagnostic smoke test:

```bash
bash scripts/smoke_1node.sh
```

For the primary two-node controlled run, `scripts/launch_2node.sh` defaults to the world16 FSDP2 config. Set `NODE_RANK` and `MASTER_ADDR` on each node as documented in [`docs/RUNBOOK_2NODE_A100.md`](docs/RUNBOOK_2NODE_A100.md).

## Experimental branches kept separate from `main`

`main` is the exact continuous-sigma implementation. More aggressive or diagnostic variants remain isolated so they cannot silently change the primary semantics:

- `agent/h3-a100-grid1000-adaln`: Grid-1000 sigma discretization + precomputed AdaLN table / parameter elimination;
- `codex/grid1000-adaln-results-20260820`: Grid-1000 cluster results and Nsight diagnostic artifacts;
- `agent/h3-a100-shared-backbone`: earlier shared-backbone development history.

Grid-1000 is an **approximate** DMD variant and is not the default exact baseline.

## Tests

```bash
bash scripts/run_unit_tests.sh
```

The suite covers exact AdaLN caching, checkpoint replay scope restoration, checkpoint-boundary CPU staging, Student/Fake role separation, world16 matched forward/backward census, global sample uniqueness, pinned-upstream contracts, configuration invariants, and topology mapping.

## Current limitations

- The CPU-first loader may still instantiate a full CPU H3 module per local rank before FSDP sharding; direct DTensor/safetensors shard loading remains follow-up work for host-memory efficiency.
- The validated exact result is one controlled outer cycle; repeated formal timing is still needed before making a speedup claim.
- The exact path keeps continuous renoise sigma, so dynamic score AdaLN output is cached per sampled sigma rather than replaced by the finite Grid-1000 lookup table.
- Checkpoint/resume is currently intended for the same topology.
