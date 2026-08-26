# H3 Grid1000 identity LoRA scale elision

The trusted Iteration 360 profile recorded 15,101 BF16 unary multiplication
kernels totaling 13.879 s/GPU. Source and config audit showed that active PEFT
LoRA paths execute `lora_B(lora_A(x)) * scaling` while both Student and Fake use
rank=alpha=128, making every controlled scaling value exactly 1.0.

The opt-in candidate pins the PEFT 0.18.1 `Linear.forward` source identity and
removes only this identity multiplication. Base and LoRA A/B GEMMs, residual
addition, adapter routing, FSDP, and all training operations remain unchanged.
It fails closed for non-unit scaling, dropout, LoRA variants, multiple active
adapters, or unsupported call arguments.

A real installed-PEFT CUDA canary at `[1,37760,5376] -> 7168`, rank128, was
bitwise equal for output, input gradient, and all parameter gradients. Its
paired median was 18.192 ms reference versus 16.639 ms candidate.

Iteration 366 completed the compute but rejected its post-operation counters by
12 grad calls. This was the known non-reentrant checkpoint early-stop behavior:
Python after the final recomputed tensor need not execute. Only the diagnostic
counter moved to path entry; Iteration 367 then passed exactly:

- 312 PEFT Linear modules;
- 11,544 total calls;
- 11,232 identity scales elided;
- 7,488 no-grad and 3,744 grad/replay elisions;
- 312 Teacher adapter-disabled reference calls;
- zero unsupported/invalid calls.

| Run | One cycle | Delta vs parent | Status |
|---|---:|---:|---|
| Iteration 365 FA3 parent | 848.128 s | -- | PASS |
| Iteration 367 LoRA scale-one | 836.298 s | -11.830 s (-1.395%) | PASS, one run |
| Iteration 327 original control | 904.986 s | -68.688 s cumulative | PASS control |

All 16 ranks passed application forward 24/6/1, grad forward 1/5/0, backward
1/5, Grid replay, and all inherited fusion gates. Peak rank0 HBM was
34.79/36.63 GiB allocated/reserved, with no CUDA/cgroup OOM and clean teardown.
The timing is a provisional single run, not a confidence interval.
