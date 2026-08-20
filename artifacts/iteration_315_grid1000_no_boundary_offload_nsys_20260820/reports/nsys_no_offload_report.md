# Nsight diagnostic: Grid1000 no-boundary-offload MiniMax-H3

- **Run ID:** `iteration_315_grid1000_no_boundary_offload_nsys_20260820`
- **Status:** `PROFILE_PARTIAL_FAIL_NOT_USABLE_WORLD16`
- **Purpose:** diagnostic only; no formal timing or speedup claim.
- **Source HEAD:** `75c837fac356a50883bed6963dc4bf96218d99c8`
- **Workload:** world16 (2 nodes × 8 A100-40GB), Grid1000, one outer cycle (Student 1 + Fake 5), BF16, B1/rank, 768×1344×124, checkpoint segment=1, `activation_policy=none`, boundary CPU offload disabled.

## Application gate

The application completed its one-cycle run. The saved rank-0 log records:

- forward census `Student/Fake/Teacher = 24/6/1`; grad-forward `1/5/0`; backward `1/5`;
- Grid replay `checkpoint_wrap_count=300`, `scoped_execution_count=600`, `cache_hit_count=600`, `cache_miss_count=0`, `missing_key_count=0`;
- `activation policy=none`, `boundary_events=False`;
- `train finished`, no CUDA OOM in this run.

This is independent of the profile validity below.

## Why the Nsight result is not a valid world16 utilization result

Nsight Systems 2023.3.1 emitted `ProcessEventsError`/`Cannot find bucket for a bucket index` and `Unknown runtime API function index: 468`. The exported SQLite is therefore incomplete:

| node | NVTX phases | kernel events in workload | devices with kernel events / 8 | unresolved-kernel union | profile processing error |
|---|---:|---:|---:|---:|---|
| node0 | 2/2 | 3853 | 2,7/8 | 21.400995 s | yes |
| node1 | 1/2 | 20 | 4/8 | 0.550968 s | yes |

The node1 SQLite contains only 20 kernel events and no `h3/critic_update_F` range; node0 contains kernel events only on devices 2 and 7. Consequently, the numbers below are **observed fragments**, not A100 SM utilization, full-world compute coverage, or total NCCL overhead.

## Observed CUDA timeline fragments

| node | workload NVTX wall | all observed kernel union | observed union / wall | known non-NCCL union | known NCCL union | known NCCL exposed vs known compute |
|---|---:|---:|---:|---:|---:|---:|
| node0 | 903.880056 s | 21.403605 s | 2.368% | 0.000528 s | 0.002082 s | 0.002082 s |
| node1 | 903.730141 s | 0.550968 s | 0.061% | 0.000000 s | 0.000000 s | 0.000000 s |

The apparent 2.368%/0.061% “coverage” must **not** be called compute utilization: it is the fraction of the recorded kernel-event union, and the recorder dropped most GPU activity. No SM Active/Tensor Active hardware metrics were collected.

Only one named NCCL kernel (`ncclDevKernel_AllReduce_Sum_bf16_TREE_LL`) is resolvable on node0 (2.081868 ms). All other NCCL/compute kernels are unresolved or missing, so an exposed-NCCL total cannot be obtained from this capture.

## H2D/D2H and offload attribution

`copyKind=2` is `CUDA_MEMCPY_KIND_DTOH` according to the exported enum. Within the captured workload NVTX ranges:

| node | H2D events / bytes | D2H events / bytes | D2H union | copy-kind-8 events / bytes |
|---|---:|---:|---:|---:|
| node0 | 72 / 20,622,304 B | 0 / 0 B | 0.000000000 s | 1 / 1,248 B |
| node1 | 30 / 501,800 B | 2 / 8 B | 0.000006368 s | 0 / 0 B |

Therefore this no-offload run shows **no activation/checkpoint-boundary D2H**: node0 has zero D2H events and node1 has only two 4-byte DTOH events (8 B total), which are diagnostic/control-sized copies, not model activation traffic. The app contract also reports `activation_policy=none` and `boundary_events=False`. H2D inside the workload ranges is only 21,124,104 B across both nodes; the larger full-capture H2D totals (2,657,115,104 B and 2,310,820,904 B) occur largely before the workload NVTX and are setup/materialization/staging, not evidence of activation offload.

## Answers

1. **Compute-kernel utilization:** not measurable from this capture. The recorded kernel-union fractions are incomplete fragments, not GPU utilization.
2. **Exposed NCCL:** not measurable for the full world16; only a single named 2.081868-ms all-reduce kernel is resolvable.
3. **Offload D2H:** no meaningful offload contribution is present; workload-range D2H is 0 B on node0 and 8 B on node1.

## Artifacts

- Raw reports: `raw/node0/*.nsys-rep`, `raw/node1/*.nsys-rep`
- SQLite: `raw/node0/*.sqlite`, `raw/node1/*.sqlite`
- Analyzer: `analysis/analyze_nsys_no_offload.py`
- Machine-readable summary: `analysis/nsys_no_offload_summary.json` and `.csv`
- Manifest: `manifests/nsys_no_offload_manifest.json`

**Next minimal action (not run):** use a newer Nsight Systems build compatible with this CUDA/PyTorch runtime, or disable the unsupported runtime/event path while retaining CUDA kernel + NVTX + memcpy capture, then rerun one diagnostic capture. Do not use this profile for timing or speedup.
