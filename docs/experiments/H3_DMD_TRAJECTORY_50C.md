# H3 DMD 50-cycle trajectory receipts

Status: **NOT_RUN**. This change adds the bounded evidence path only; it does
not contain a GPU result.

`H3_TRAJECTORY_MODE=coarse_v1` is an explicit diagnostic workload used to
compare a completed Exact run with a completed Grid-1000 run. It forces 50
outer cycles and keeps the existing per-cycle matched-compute and replay gates.
There is no online numerical early stop.

The two arms use a shared operation-keyed continuous renoise draw for each
`rank/cycle/{student,fake_0..fake_4}`. Exact consumes that continuous value;
Grid-1000 snaps the same value to the nearest point of the frozen 1000-point
grid. The sampler is independent of the global PyTorch RNG, so it does not
shift the latent/noise RNG stream between arms. This trajectory fixture is not
interchangeable with older native-RNG timing results.

Each rank writes an append-only JSONL containing:

- six sample identities and the existing 24/6/1 forward, 1/5/0 grad-forward,
  and 1/5 backward census per cycle;
- one Student loss and five ordered Fake losses;
- continuous and actual sigma plus Grid snap receipt;
- logical optimizer/scheduler/EMA update versions;
- bounded parameter-delta, optimizer-state and post-clip gradient sketches at
  cycles 1, 10, 25 and 50.

Launch each branch from a fresh evidence directory on both hosts:

```bash
export H3_EXPECTED_HEAD=$(git rev-parse HEAD)
export H3_TRAJECTORY_DIR=/persistent/path/<new-run-id>/receipts
export NODE_RANK=0  # 1 on peer
export MASTER_ADDR=<node0-ip>
export MASTER_PORT=<new-port>
scripts/launch_trajectory_50c.sh exact
```

Use `grid1000` on the Grid winner branch and provide the already validated
`H3_ADALN_GRID_MANIFEST`. After both complete:

```bash
python scripts/compare_h3_trajectories.py \
  --reference /persistent/path/exact/receipts \
  --candidate /persistent/path/grid1000/receipts \
  --expected-cycles 50 \
  --output /persistent/path/comparison/exact_vs_grid1000.json
```

The `0.999` cosine and loss-curve thresholds are soft final labels only. Every
50-cycle run completes unless an existing semantic/census/replay gate or a
runtime failure stops it.
