# Iterations 400-418 — pageable-staged exact FA3 replay cache

## Controlled result

The exact world16 Grid1000 workload reached **772.332 s** in Iteration 418.
This is one unprofiled matched run, not a three-run variance claim.

| Run | Replay policy | Blocks | In-flight pinned stages | Wall | Status |
|---|---|---:|---:|---:|---|
| Iteration 393 | no FA3 replay cache | 0 | 0 | 812.029 s | PASS baseline |
| Iteration 416 | pageable reservoir + staged copies | 11-49 | 1 | 788.806 s | PASS |
| Iteration 417 | pageable reservoir + staged copies | 11-49 | 2 | 782.591 s | PASS |
| Iteration 418 | pageable reservoir + staged copies | 0-49 | 2 | **772.332 s** | **PASS target** |

Iteration 418 improves the matched control by **39.697 s (4.89%), 1.05140x**
and finishes **7.668 s** below the requested 780 s target.

## Frozen workload contract

- 2 nodes x 8 A100-40GB, world16 FSDP2 FULL_SHARD
- MiniMax-H3 Grid1000, BF16, B1/rank and global batch 16
- one outer cycle: Student1 + Fake5
- application forwards: Student/Fake/Teacher = 24/6/1
- grad forwards: 1/5/0; backwards: 1/5
- checkpoint segment 1; activation offload disabled
- Grid replay: 300 wraps, 600 scoped executions, 600 hits, 0 misses
- model, prompt cache, RNG, optimizer, topology, and mathematical schedule unchanged

## Mechanism and bounded search

The earlier raw-pinned implementation was rejected after reproducing the known
host/capacity failure. The accepted implementation instead uses:

1. a long-lived **pageable** CPU reservoir for cached FA3 output and LSE;
2. a bounded reusable pinned staging window for physical D2H/H2D transfers;
3. asynchronous pageable-to-pinned prefetch before checkpoint replay;
4. exact rank-local dynamic sequence views inside max-shape reservoirs;
5. allocator trimming only at the six pre-backward boundaries.

The search remained one-variable at each world16 step. Iteration 416 established
the dynamic-shape path at depth 1. Iteration 417 changed only the pinned staging
depth from 1 to 2. Iteration 418 then changed only the selected block range from
11-49 to 0-49.

## Iteration 418 receipts

- exact source: `cfecbb4188e1f3820451657f744aa2c823e24da4`
- focused FA3 replay tests: 8/8 on each node; full suite: 74/74 on node0
- selected checkpoint wraps / scoped executions: 300 / 600
- compact forward / backward calls: 300 / 300
- rank0 bytes on each reconciled path: 164,794,448,000 B
  - logical cached bytes
  - CUDA D2H bytes
  - CUDA H2D bytes
  - forward pinned-to-pageable bytes
  - backward pageable-to-pinned bytes
- backward prefetches / misses: 300 / 0
- unexpected storage/kernel/checkpoint contract calls: 0 / 0 / 0
- rank0 peak allocated: 33.39 GiB
- minimum driver free: node0 998 MiB, node1 128 MiB
- cgroup peak: node0 239.60 GiB, node1 239.48 GiB
- CUDA OOM: 0; cgroup OOM-kill delta: 0/0
- node launchers: 0/0; post-run GPU processes: 0; all locks free

The node1 HBM margin was only 128 MiB. The run is an actual capacity PASS under
the explicitly authorized actual-OOM boundary, but it must not be described as
having a robust reserve margin.

## Evidence

Persistent raw evidence remains outside Git under:

`/home/lyy/Helios/research_artifacts/H3-DMD-for-A100/iteration_418_fa3_pageable_staged_all50_inflight2_20260827/`

It includes the immutable launchers, preflight, both node logs and monitoring
CSVs, exact result JSON, report, and SHA256 manifest.
