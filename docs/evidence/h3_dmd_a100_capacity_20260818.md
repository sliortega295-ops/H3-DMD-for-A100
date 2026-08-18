# H3-DMD-for-A100 capacity evidence (2026-08-18)

This archive contains the preserved preflight, smoke, world16 segmented-checkpoint,
and selective activation-offload evidence for the bounded MiniMax-H3 A100 run.
It does not contain model weights or prompt tensors.

Source:
- H3-DMD-for-A100 HEAD: `1dd4a7b607b5a575d10e7bc17564251c7ac72979`
- LightX2V: `d034a6b0ecaa31aa07c81aeb7bbe69b225f1d7be`
- Workload: 2 nodes x 8 A100-40GB, world16, BF16, 768x1344x124,
  Student/Fake LoRA rank 128, fixed end_step_idx=3, Student1/Fake5.

Result:
- One-node smoke paths passed.
- World16 segment=1, segment=8, segment=8+activation-offload-128 MiB,
  and segment=8+activation-offload-64 MiB all failed at the first Student
  backward/recompute capacity point.
- No Full5 timing or speedup was claimed.
- CP/SP2 remains NOT_RUN because a semantics-preserving MiniMax-H3 integration
  is not available in the pinned LightX2V path.

The JSON/report in the repository is the concise status source; the adjacent
archive is the raw bounded evidence bundle.
