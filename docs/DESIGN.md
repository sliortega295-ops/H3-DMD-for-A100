# Design: shared-backbone H3 DMD

## 1. Baseline execution graph

LightX2V's H3 trainer inherits the generic DMD loop and creates independent Student, Fake, and Teacher model objects. Each object loads its own transformer and is FSDP-wrapped independently. For H3 LoRA-DMD, the three base checkpoints are numerically equal; only the Student and Fake adapters evolve.

The optimized graph retains the DMD mathematics while changing physical ownership:

```text
physical module: one H3 transformer
trainable sets:  student LoRA, fake LoRA
logical views:   student adapter, fake adapter, adapters disabled
optimizers:      independent AdamW states
```

The graph remains sequential. The design does not attempt concurrent Student/Fake/Teacher forwards through the same module.

## 2. Shared-backbone invariants

The following conditions must hold:

1. Student and Fake use the same frozen base checkpoint.
2. Teacher is exactly the base checkpoint with all adapters disabled.
3. Student and Fake optimizer parameter sets are disjoint.
4. Adapter selection is restored around both forward and gradient-checkpoint recomputation.
5. Both named adapter sets remain marked trainable before FSDP wrapping; active-adapter selection determines which set participates in a forward.

`MiniMaxH3A100Model.role_scope()` enforces role selection. The trainer re-enters the correct role scope around `.backward()` because activation checkpointing replays block forwards during backward.

## 3. AdaLN bank and cache

### 3.1 Why extraction is necessary

Simply memoizing `adaln_proj(temb)` inside an original FSDP-wrapped block does not remove the AdaLN weights from the block all-gather. The implementation therefore:

1. removes every `block.adaln_proj` module;
2. registers the 50 projections once under `h3_a100_adaln_bank`;
3. installs parameter-free weak-reference handles in the blocks;
4. makes each projection a separate FSDP unit.

A cached block forward then bypasses both the AdaLN matmul and that projection's parameter all-gather.

### 3.2 Cache keys

Two classes of keys are used:

- `("rollout", step_idx)`: persistent, because the four Student rollout sigmas are fixed for the complete run;
- `("score", serial)`: dynamic, because DMD samples a continuous renoise sigma.

For a Student update, Fake and Teacher evaluate the same renoised latent at the same video/audio sigma pair, so both roles share one dynamic table. For a Fake update, the dynamic table is retained through backward so activation-checkpoint replay does not recompute AdaLN.

### 3.3 Exactness contract

The cache stores the original module outputs in the original dtype. It detaches them because:

- `adaln_proj` is frozen;
- `time_embedder` is frozen;
- the configured LoRA targets are only attention and FFN projections.

If a future configuration trains AdaLN or the timestep MLP, installation raises an error rather than silently serving stale values.

## 4. Critic rollout reordering

Let `G_v` be the fixed Student version after the Student optimizer step, and `F_i` the Fake model after the `i`-th Fake update.

Baseline:

```text
x1 ~ G_v; F_1 = Update(F_0, x1)
x2 ~ G_v; F_2 = Update(F_1, x2)
...
x5 ~ G_v; F_5 = Update(F_4, x5)
```

Reordered:

```text
x1,...,x5 ~ G_v
F_1 = Update(F_0, x1)
F_2 = Update(F_1, x2)
...
F_5 = Update(F_4, x5)
```

The optimizer trajectory is unchanged because every rollout depends on `G_v` but not on `F_i`. H3's block dropout is zero, so moving Fake forwards after all rollouts does not interleave an additional model RNG stream. Five optimizer steps remain five optimizer steps.

The buffered objects are generated latents, renoise noise/sigmas, prompt embeddings, and layout metadata. They are not concatenated into one batch.

## 5. HSDP and input synchronization

With a 2×8 mesh, ranks 0–7 form replica 0's shard group and ranks 8–15 form replica 1's shard group. Every rank in a shard group reconstructs and executes the same model forward, so inputs must be identical inside that group.

The overlay changes LightX2V's data-parallel rank/world to the replicate dimension. Thus all eight ranks in one node receive the same sample. Random end steps, initial video/audio noise, renoise sigma, and renoise noise are drawn only on shard rank 0 and broadcast to the other seven ranks. Replicas use different seeds and samples.

## 6. Initialization and host memory

The upstream wrapper calls `.to(cuda)` before FSDP. The overlay leaves the loaded module on CPU. FSDP2 converts CPU parameters into DTensor shards on the CUDA device mesh. This avoids a transient 61.7-GB transformer on one 40-GB GPU.

This is not yet a zero-copy host loader. Eight local processes still instantiate full CPU modules before sharding. The implementation calls `gc.collect()` and `malloc_trim(0)` after FSDP, but the node must have enough RAM for the initialization peak.

A future version should load safetensor shards directly into meta/FSDP parameters or share read-only checkpoint pages across local ranks.

## 7. Checkpoint format

The frozen base is reloaded from `MINIMAX_H3_MODEL_PATH` on resume. Checkpoints contain:

```text
student/pytorch_lora_weights.safetensors
fake/pytorch_lora_weights.safetensors
rank-0000.pt ... rank-0015.pt
trainer_state.pt
_SUCCESS
```

The two adapter files are gathered in one FSDP state-dict operation with frozen parameters ignored. Each rank stores its local optimizer/scheduler/RNG state. This is intentionally a same-world-size format for the first cluster implementation.

## 8. Deferred optimizations

These should be evaluated only after semantic and numerical validation:

- direct sharded/meta checkpoint loading to lower host-memory peak;
- activation offload for the final differentiable Student/Fake block path;
- optimizer-state CPU offload if adapter rank remains 128;
- context/sequence parallelism for larger canvases;
- `torch.compile` after adapter switching and FSDP graphs stabilize;
- quantized frozen base weights, provided LoRA training kernels and DMD accuracy remain valid;
- caching the much smaller final `norm_out` timestep projection.
