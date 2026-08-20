## Iteration 308 — shared-backbone segment1 boundary-policy block attribution

- Question: does the proven `checkpoint_boundary_cpu`/segment1 policy fit the shared-backbone MiniMax-H3 world16 path?
- Hypothesis: with one physical base and per-block boundary staging, Student1+Fake5 should fit under 40 GiB.
- Result: **FAIL_CUDA_OOM_DURING_STUDENT_BACKWARD_RECOMPUTE**. Student grad forward completed; first backward recompute failed at LoRA FFN with a 2.02-GiB request while PyTorch held 36.79 GiB.
- Boundary evidence: 50 copies / 20,283,648,000 B before backward; formal 300-copy census not reached.
- Timing/FULL5: NOT_RUN.
- Interpretation: backward recompute transient is proven; exact delta vs DMD-System remains unresolved.
- Next question: can H3 use the DMD checkpoint metadata-tolerant/reentrant-compatible wrapper without changing math?
