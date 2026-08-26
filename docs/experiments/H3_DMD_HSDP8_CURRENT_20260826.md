# Iteration 391: current optimized HSDP8 no-offload

## Result

**Capacity/correctness PASS, performance REJECT.** The current kernel stack makes the formerly failing HSDP topology fit without activation offload, but it is slower and uses more HBM than the matched world16 FSDP2 parent.

| Variant | Mesh | One-cycle wall | Rank-0 peak allocated | Status |
|---|---|---:|---:|---|
| Iteration 390 parent | world16 FULL_SHARD | 813.609 s | 33.39 GiB | PASS |
| Iteration 391 | replicate=2, shard=8 HSDP | 827.867 s | 36.22 GiB | PASS, performance reject |

The topology-only candidate is **14.258 s slower (+1.752%)** and adds **2.83 GiB/rank** at the observed rank-0 peak. Its setup residency is 5.01 GiB versus the prior world16 control's 2.50 GiB.

## Gates

- Exact source: `8c3fc66bf8b9b6b36225034edb5ad0f664233969`.
- HSDP receipts: 8 ranks on each node, `world=16 replicate=2 shard=8`, while data-parallel semantics remain world16.
- Application census: 24/6/1 forwards, 1/5/0 grad-forwards, 1/5 backwards.
- Grid replay: 300 wraps, 600 scoped executions, 600 cache hits, 0 misses.
- 96 rank-qualified sample identities; all current fusion/LoRA counters passed.
- No CUDA or cgroup OOM; both launchers exited 0; postflight found 16 idle GPUs and all three locks free.

## Interpretation boundary

This single unprofiled run is sufficient to reject HSDP as the next performance winner under the current setup: node-local 8-way parameter collectives do not compensate for doubled local shard residency and the associated memory pressure. It is not a variance-qualified formal timing result. We retain the 813.609 s world16 FSDP2 parent. No offload, workload, precision, schedule, or optimizer change was introduced.
