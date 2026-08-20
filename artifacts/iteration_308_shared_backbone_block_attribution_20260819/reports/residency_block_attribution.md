# Iteration 308 — Shared-backbone boundary-policy block attribution

- **Run ID:** `iteration_308_shared_backbone_block_attribution_20260819`
- **Status:** **FAIL — CUDA OOM during Student backward checkpoint recompute**
- **Source:** `H3-DMD-for-A100` `316d2ad38b395ee8547840ca299181f34ecb860e` (`agent/h3-a100-shared-backbone`)
- **Upstream LightX2V:** `d034a6b0ecaa31aa07c81aeb7bbe69b225f1d7be`
- **Timing / FULL5:** **NOT RUN** (no performance claim)

## Contract and policy

This run used the unchanged full workload: world16 (2×8 A100-40GB), 768×1344×124, BF16, B1/rank, Student/Fake LoRA rank128, shared frozen Teacher, four-step rollout with `end_step_idx=3`, Student1 + Fake5, `_flash_3_hub`. The only runtime policy was `checkpoint_segment=1` with `checkpoint_boundary_cpu` and pinned boundary storage; parameter offload, model swap, CP/SP, segment8 and threshold offload were disabled.

The two code commits relative to `b7508ea1ea8f1909545ded15ae2be314f6afa093` only add opt-in residency/block attribution (`H3_MEMORY_ATTRIBUTION*`); they do not change loss, optimizer, model, trace or schedule semantics.

## Outcome

Construction and Student gradient forward completed. The first Student backward entered checkpoint recompute and failed on a 2.02-GiB allocation in `peft/tuners/lora/layer.py:807` (LoRA FFN projection). All 16 global ranks failed with the same CUDA OOM signature (the captured text has 17 records because one launcher/root record duplicates a rank failure). No Fake update, optimizer commit, application census or formal boundary census completed; Full5 and timing are **NOT_RUN**.

Representative rank-0 memory (binary GiB):

| point | allocated | reserved | driver free | evidence |
|---|---:|---:|---:|---|
| before Student grad forward | 4.02 | 4.07 | 34.58 | static setup |
| after Student grad forward | 5.01 | 18.59 | 19.99 | 50 boundary copies |
| before Student backward | 5.01 | 18.59 | 19.99 | recompute about to start |
| deepest confirmed block-9 post | 37.07 | 38.50 | 0.07 | block hook |
| minimum observed driver free | 37.07 | 38.50 | 0.07 | `block_9_post` |
| OOM allocation request | 36.79 PyTorch allocated | ~37.66 process | 1.63–1.67 | attempted 2.02 GiB |

Boundary staging was active: the last completed forward snapshot reports **50 CPU boundary copies**, **20,283,648,000 B (18.89 GiB)** logical/storage bytes, `pack_count=200`, `unpack_count=0`. This is partial pre-backward evidence, not a completed 300-copy cycle census.

## Block traversal evidence

The representative log contains six complete 0–49 block traversals (the six grad/application traversals) before backward and then a seventh, descending backward-recompute traversal that is partial (49 down to 8). The deepest confirmed post-hook before termination is block 9 (about 37.07 GiB allocated); block 8 post is the last later hook recorded (about 34.78 GiB), and the traceback occurs during a subsequent checkpoint unpack/recompute LoRA projection. Therefore the exact logical block of the failed allocation is **UNRESOLVED**; it must not be reported as a precise block number.

## Comparison with the successful DMD-System reference

| path | peak allocated | wall | status |
|---|---:|---:|---|
| DMD-System Native (frozen result) | 27.94 GiB | 917.993973 s | PASS |
| DMD-System phase-cluster (frozen result) | 28.10 GiB | 914.102536 s | PASS |
| H3 shared backbone + segment1 boundary CPU | setup 4.02 GiB; backward transient >37 GiB | — | FAIL CUDA OOM |

The static shared-backbone setup is not larger because it has more frozen weights: only one physical base is instantiated. The measured failure is a backward-recompute transient (FSDP unit materialization + activation/LoRA workspace/allocator pressure). The exact ~9–10 GiB difference from the DMD reference is not yet isolated, so it is **not** valid to call it a model-offload or parameter-residency issue.

## Evidence boundary

**PROVEN**

- Segment-1 boundary policy was selected and boundary hook ran.
- Static setup and full Student gradient forward fit.
- All 16 global ranks reached the same failure class; no rank completed the outer cycle.
- Boundary CPU staging reached 50 copies/18.89 GiB before backward.
- CUDA OOM happened in Student backward checkpoint recompute at the LoRA FFN projection.
- Node cgroup `oom` did not increase; the pre-existing node0 `oom_kill=4` is historical and unchanged.

**STRONG INFERENCE**

- The dominant failing region is the backward recompute transient, not the static shared backbone alone.
- A completed forward with boundary copies does not guarantee enough HBM for recompute because FSDP all-gather/materialization and LoRA projection workspace coexist at the peak.

**UNRESOLVED**

- Exact residency delta versus DMD-System; checkpoint context/metadata behavior, shared-adapter lifecycle, AdaLN/FSDP resharding, allocator fragmentation, and offloader implementation differences remain to be isolated.
- Exact block of the failed allocation.

## Artifacts and cleanup

- Representative raw logs: `logs/node0/output_train.log`, `logs/node0/train.log`, `logs/node1/train.log`
- Machine-readable rows: `reports/residency_block_attribution.csv`
- Machine-readable summary: `reports/residency_block_attribution.json`
- No GPU workers remained after teardown; no new cgroup OOM was observed.

**Next bounded action (not run here):** source-parity audit of H3 checkpoint wrapper against the already successful DMD `fsdp2_metadata_tolerant_checkpoint` / boundary hook, followed by a single Student backward canary. Do not jump to segment8, threshold offload, CP/SP or a timing run.
