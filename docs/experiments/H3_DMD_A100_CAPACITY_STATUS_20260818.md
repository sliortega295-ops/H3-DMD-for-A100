# MiniMax-H3 A100 capacity status (bounded attempts)

This is an interim capacity report, not a Full5 performance result.  Formal
timing is **NOT_RUN** because no candidate has yet passed the exact world16
one-cycle capacity/correctness gate.

## Frozen contract

- 2 nodes x 8 A100-40GB, world16, BF16
- MiniMax-H3 video+audio at 768x1344 and 124 frames
- Student LoRA rank 128, Fake LoRA rank 128, one shared physical backbone
- fixed `end_step_idx=3`, continuous renoise sigma, `_flash_3_hub`
- one Student update followed by five ordered Fake updates, GAS=1, B1/rank
- application census target: Student/Fake/Teacher = 24/6/1 forwards;
  grad-enabled = 1/5/0; backward = 1/5
- no loss, optimizer, sample, RNG, precision, or topology changes

## Source and environment identity

- H3-DMD-for-A100 branch: `agent/h3-a100-shared-backbone`
- current source HEAD: `f00ef38406fcfeea06e45567482ecc898df192aa`
- pinned LightX2V: `d034a6b0ecaa31aa07c81aeb7bbe69b225f1d7be`
- Torch: 2.10.0+cu128; CUDA runtime 12.8; driver 535.161.08
- Diffusers: 0.40.0.dev0, source `9284607295a09f759aadd65ed08f48b35feea6d9`
- Transformers 4.57.6; PEFT 0.18.1; local pinned FlashAttention-3
- model and prompt-cache identities are those recorded in
  `iteration_301_preflight_20260818`

## Gates and runs

| Run | Candidate | Scope | Status | Evidence |
|---|---|---|---|---|
| `h3_smoke_1node_shared_backbone_20260818_r12_fsdp_root_init` | one-node shared backbone, per-block checkpoint | one smoke cycle | **PASS** | exact census, no OOM, cgroup delta 0 |
| `h3_smoke_1node_shared_backbone_20260818_r13_sac_seg8` | segment=8 | launcher/import | **FAIL_LAUNCHER** | wrong Python environment (`huggingface_hub` missing) |
| `h3_smoke_1node_shared_backbone_20260818_r14_sac_seg8` | segment=8 | one-node smoke | **FAIL_CUDA_OOM** | AdaLN projection replay, scope not preserved |
| `h3_smoke_1node_shared_backbone_20260818_r15_sac_seg8_scopefix` | segment=8 | one-node smoke | **FAIL_CUDA_OOM** | same replay scope issue remained |
| `h3_smoke_1node_shared_backbone_20260818_r16_sac_seg8_scopefallback` | segment=8 | one-node smoke | **PASS** | exact census, max allocated 29.36 GiB, cgroup delta 0 |
| `h3_world16_shared_backbone_onecycle_20260818_r2_fsdp_root_init` | segment=1 | world16 one cycle | **FAIL_CUDA_OOM** | first Student backward recompute, 2.02 GiB request |
| `h3_world16_shared_backbone_onecycle_20260818_r3_sac_seg8` | segment=8 | world16 one cycle | **FAIL_CUDA_OOM** | first Student backward recompute, 388 MiB request |
| `h3_world16_shared_backbone_onecycle_20260818_r4_sac_seg8_actoff128` | segment=8 + saved-tensor offload >=128 MiB | world16 one cycle | **FAIL_CUDA_OOM** | FFN LoRA projection during checkpoint replay, 2.02 GiB request |
| `h3_world16_shared_backbone_onecycle_20260818_r5_sac_seg8_actoff64` | segment=8 + saved-tensor offload >=64 MiB | world16 one cycle | **FAIL_CUDA_OOM** | same FFN LoRA projection, 2.02 GiB request |

## Selective activation offload evidence

The opt-in implementation is thresholded, applies only to grad-path saved
tensors, excludes parameters/buffers, and disables additional offload during
checkpoint replay.  It does not change application-level forward/backward
counts.  Focused tests on node0: 10 passed.

The rank-0 receipt from r5 shows that the hook was active and copied 9 saved
tensors (3,651,056,640 bytes, about 3.40 GiB) to CPU.  Despite that, every
world16 rank reached the same first Student backward replay failure:

```
attempted allocation: 2.02 GiB
allocated:             about 37.87 GiB
reserved-unallocated:  about 644 MiB
driver free:           74--134 MiB
location:              FSDPMiniMaxH3TransformerBlock FFN LoRA projection
cgroup OOM delta:      0 (node0 oom_kill counter unchanged; node1 unchanged)
```

This is a bounded negative result for thresholded saved-tensor offload at
128 MiB and 64 MiB.  The failing allocation is a live checkpoint-recompute
FFN workspace/LoRA projection, not a saved tensor that the hook can release;
lowering the threshold did not move the failure point or reduce peak HBM.

## Current gate conclusion

`PRIMARY_CAPACITY_PASS`: **NOT_RUN** (world16 candidate still fails).

`SAC_CAPACITY_PASS`: **FAIL** at world16.

`SELECTIVE_ACTIVATION_OFFLOAD_128`: **FAIL**.

`SELECTIVE_ACTIVATION_OFFLOAD_64`: **FAIL**.

No Full5 timing, speedup, Nsight, or training-quality claim is made.

The next preregistered mechanism is context/sequence parallelism (size 2),
but the pinned H3/LightX2V path contains only generic sequence-parallel helper
utilities; MiniMax-H3 attention does not use them.  Implementing a correct
H3 CP/SP path would require a separate semantics-preserving design and is not
safe to substitute with a broadcast or duplicated sample.  Therefore CP/SP
is currently **NOT_RUN**, not silently approximated.

## Raw evidence

- r3: `iteration_302_sac_seg8_world16_20260818/`
- r4/r5: `iteration_303_selective_activation_offload_20260818/`
- preflight/smoke: `iteration_301_preflight_20260818/` and the shared remote
  H3 run directories named above

