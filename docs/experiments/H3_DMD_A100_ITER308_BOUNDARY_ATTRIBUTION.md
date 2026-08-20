# H3-DMD-for-A100 Iteration 308: segment-1 boundary-policy attribution

This is a diagnostic/capacity-failure report, not a timing result.

- Source branch: `agent/h3-a100-shared-backbone`
- Source HEAD: `316d2ad38b395ee8547840ca299181f34ecb860e`
- Workload: world16, 2×8 A100-40GB, MiniMax-H3 768×1344×124, BF16, B1/rank, Student1 + Fake5
- Policy: `checkpoint_segment=1`, `checkpoint_boundary_cpu`, pinned boundary storage; no segment8, threshold offload, parameter offload, CP/SP, or model swap
- Result: **FAIL_CUDA_OOM_DURING_STUDENT_BACKWARD_RECOMPUTE**
- Timing / Full5: **NOT_RUN**

The representative rank completed Student gradient forward and recorded 50 boundary copies (20,283,648,000 B), then failed in the first backward checkpoint recompute at a PEFT LoRA FFN projection while requesting 2.02 GiB. Peak confirmed allocation was 37.07 GiB, reserved 38.50 GiB, with about 0.07 GiB driver free. All 16 global ranks failed with the same CUDA OOM class; cgroup OOM did not increase.

The detailed machine-readable rows, JSON summary, plot, raw logs, command and preflight receipts are in [`artifacts/iteration_308_shared_backbone_block_attribution_20260819/`](../../artifacts/iteration_308_shared_backbone_block_attribution_20260819/).

This result does **not** establish that static model weights or parameter offload are the cause. The exact 9–10 GiB gap versus the successful DMD-System reference remains unresolved; the next bounded step is checkpoint-wrapper/FSDP lifecycle parity, not segment8.
