# Validation and ablation plan

The first goal is not to claim speedup; it is to establish semantic correctness and a stable two-node baseline. Run the following gates in order.

## A. Static and CPU unit tests

```bash
bash scripts/run_unit_tests.sh
```

Covered invariants:

- frozen AdaLN required;
- persistent cache is exact and does not recompute;
- dynamic cache is reused and evicted correctly;
- 2×8 rank mapping;
- pinned upstream/config/script contracts.

## B. One-node functional matrix

Use the smoke canvas and one iteration.

| Variant | Shared backbone | AdaLN cache | 5G→5F reorder | Purpose |
|---|---:|---:|---:|---|
| B0 | yes | off | off | shared-backbone control |
| B1 | yes | on | off | isolate AdaLN cache |
| B2 | yes | on | on | complete optimized control flow |

For the first implementation, toggles can be changed in a copied YAML. Use the same prompt cache and seed.

Record:

- Student/Fake losses;
- Student/Fake LoRA gradient norms;
- generated latent checksums before the first optimizer step;
- peak GPU and host memory;
- wall time per Student phase and critic phase;
- AdaLN cache hits/misses/bytes.

Expected numerical relation:

- cache on/off should be bitwise or extremely close in the same dtype because cached tensors are outputs of the original frozen projections;
- reorder on/off should preserve the five optimizer-step semantics, but floating-point execution order and asynchronous collectives may produce small differences over multiple iterations.

## C. HSDP correctness

Compare one-node 1×8 and two-node 2×8:

- ranks inside each shard group must log identical end-step/sigma/noise checksums;
- the two replicas should receive different dataset samples;
- gradients must reduce across replicas;
- losses should equal the average of the two replica samples, within distributed floating-point tolerance.

## D. Full-resolution short run

Run 3–10 iterations at 1344×768×124 before a long job. Validate checkpoint save/resume and capture an Nsight Systems trace.

## E. Performance ablations

| ID | Topology | AdaLN | Reorder | Metric of interest |
|---|---|---|---|---|
| P0 | 16-way 1-D FSDP | on | on | inter-node parameter all-gather cost |
| P1 | 2×8 HSDP | on | on | preferred topology |
| P2 | 2×8 HSDP | off | on | AdaLN compute/communication contribution |
| P3 | 2×8 HSDP | on | off | role-locality contribution |

Report:

- samples or iterations per hour;
- Student and five-Fake phase wall times;
- NCCL bytes/time split by intra-node vs inter-node transport;
- GPU active time and memory bandwidth;
- peak allocated/reserved CUDA memory;
- host-memory peak during initialization and steady state.

## F. Follow-on work after P1 is stable

1. Direct safetensor-to-DTensor/meta loading to remove eight full CPU copies per node.
2. CPU optimizer-state offload if rank-128 LoRA optimizer memory is material.
3. Activation offload only for the final differentiable rollout step and Fake forward.
4. Context parallelism for longer clips/canvases.
5. Quantized frozen backbone with BF16 LoRA path.
