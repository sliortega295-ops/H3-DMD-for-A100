"""Training loop, critic rollout reordering, matched census, and diagnostics."""

from __future__ import annotations

import contextlib
import ctypes
import gc
import os
from collections.abc import Iterator

import torch
import torch.distributed as dist
from loguru import logger

from lightx2v_train.runtime.distributed import (
    barrier,
    get_rank,
    get_world_size,
    is_main_process,
    reduce_mean,
)
from lightx2v_train.runtime.parallel import set_parallel_gradient_sync
from lightx2v_train.runtime.sequence_parallel import sync_sequence_parallel_gradients

from .matched_contract import FAKE_ROLE, STUDENT_ROLE, validate_global_snapshots
from .model import FAKE_ADAPTER, STUDENT_ADAPTER
from .trajectory import TrajectoryRecorder, reset_trajectory_rng
from .trainer_runtime import PreparedFakeUpdate


class H3A100LoopMixin:
    """Optimized one-Student/five-Fake DMD scheduling."""

    def train(self):
        resume_ckpt_path, current_iter = self._resolve_resume_a100()
        self.setup(resume_ckpt_path=resume_ckpt_path)
        if is_main_process():
            os.makedirs(self.output_train_dir, exist_ok=True)
        barrier()

        grad_accum_iters = max(1, int(self.gradient_accumulation_iters))
        logger.info(
            "[h3-a100] train start iter={}/{} world={} grad_accum={} fake_ratio={} reorder={}",
            current_iter,
            self.max_train_iters,
            get_world_size(),
            grad_accum_iters,
            self.fake_update_ratio,
            self.reorder_critic_rollouts,
        )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        base_seed = int(
            self.training_config.get(
                "seed", os.environ.get("H3_BENCHMARK_SEED", "20260817")
            )
        )
        self.trajectory_recorder = TrajectoryRecorder.from_environment(
            rank=get_rank(),
            world_size=get_world_size(),
            seed=base_seed,
            variant=self.trajectory_variant,
        )
        if self.trajectory_recorder is not None:
            if not self.matched_compute_enabled:
                raise RuntimeError("H3 trajectory mode requires matched_compute.enabled=true")
            # Setup differs materially between Exact and Grid. Reset only in
            # explicit trajectory mode, after setup and before constructing or
            # consuming the sample iterator, so both arms start from the same
            # rank-qualified Python/NumPy/CPU/CUDA RNG state.
            rng_reset = reset_trajectory_rng(seed=base_seed, rank=get_rank())
            self._trajectory_student_optimizer_updates = 0
            self._trajectory_fake_optimizer_updates = 0
            self.trajectory_recorder.start_run(
                current_iter=current_iter,
                max_train_iters=self.max_train_iters,
                named_parameters=self.shared_model.denoiser_module().named_parameters(),
                student_parameters=self.trainable_params,
                fake_parameters=self.fake_trainable_params,
                rng_reset=rng_reset,
            )
        samples = self._iter_train_samples()

        while current_iter < self.max_train_iters:
            cycle_index = current_iter
            if self.trajectory_recorder is not None:
                self.trajectory_recorder.begin_cycle(cycle_index)
            if self.matched_compute_enabled:
                self.matched_cycle_census.reset()
            self._begin_boundary_offload_cycle()
            with self._nvtx("h3/student_step"):
                running_dmd = self._student_step(samples, grad_accum_iters, current_iter)
            with self._nvtx("h3/critic_phase"):
                running_fake = self._fake_steps(samples, grad_accum_iters)
            matched_snapshot = None
            if self.matched_compute_enabled:
                matched_snapshot = self._validate_matched_cycle(current_iter)
            self._validate_boundary_offload_cycle(current_iter)

            current_iter += 1
            display_dmd = reduce_mean(running_dmd)
            display_fake = reduce_mean(running_fake)
            if self.trajectory_recorder is not None:
                if matched_snapshot is None:
                    raise RuntimeError("trajectory cycle has no matched-compute snapshot")
                self.trajectory_recorder.finish_cycle(
                    cycle=cycle_index,
                    world_dmd=display_dmd,
                    world_fake=display_fake,
                    matched_snapshot=matched_snapshot,
                    student_parameters=self.trainable_params,
                    fake_parameters=self.fake_trainable_params,
                    student_optimizer=self.optimizer,
                    fake_optimizer=self.fake_optimizer,
                    student_scheduler_steps=int(
                        self.lr_scheduler.state_dict().get("last_epoch", current_iter)
                    ),
                    fake_scheduler_steps=int(
                        self.fake_lr_scheduler.state_dict().get(
                            "last_epoch", current_iter * self.fake_update_ratio
                        )
                    ),
                    student_optimizer_updates=self._trajectory_student_optimizer_updates,
                    fake_optimizer_updates=self._trajectory_fake_optimizer_updates,
                )
            if current_iter == 1 or current_iter % self.train_log_every_iters == 0:
                logger.info(
                    "[h3-a100] iter={}/{} dmd={:.6f} fake={:.6f} lr={:.8f}",
                    current_iter,
                    self.max_train_iters,
                    display_dmd,
                    display_fake,
                    self.lr_scheduler.get_last_lr()[0],
                )
                self.log_metrics(
                    {
                        "train/dmd": display_dmd,
                        "train/fake": display_fake,
                        "train/lr": self.lr_scheduler.get_last_lr()[0],
                    },
                    step=current_iter,
                )
                self.shared_model.log_adaln_stats()
                self._log_cuda_memory(f"iter_{current_iter}")

            if self.save_every_iters and current_iter % self.save_every_iters == 0:
                self.save_checkpoint_a100(current_iter)

        if self.trajectory_recorder is not None:
            self.trajectory_recorder.finish_run()
        logger.info("[h3-a100] train finished iter={}/{}", current_iter, self.max_train_iters)

    def _student_step(self, samples: Iterator, grad_accum_iters: int, current_iter: int) -> float:
        self.optimizer.zero_grad(set_to_none=True)
        running_dmd = 0.0
        for micro_idx in range(grad_accum_iters):
            sample = next(samples)
            if self.matched_compute_enabled:
                self.matched_cycle_census.note_sample("student", sample)
            conditions = self._encode_conditions(sample)
            latent_shape = self._latent_shape(sample)
            self._set_shared_gradient_sync(micro_idx == grad_accum_iters - 1)
            self._log_residency("before_student_grad_forward")
            with self._activation_offload_scope("student") as offload:
                result = self.forward_student_loss(latent_shape, conditions, current_iter=current_iter)
                self._log_residency("after_student_grad_forward")
                if offload is not None:
                    # Legacy thresholded offload is diagnostic only.  The
                    # production checkpoint-boundary policy is installed at
                    # the transformer's native checkpoint function instead.
                    offload.begin_backward()
                self.shared_model.transformer.train()
                with self.shared_model.role_scope(STUDENT_ADAPTER), self.shared_model.adaln_scope(
                    result["backward_key"], persistent=True
                ):
                    if self.matched_compute_enabled:
                        self.matched_cycle_census.note_backward(STUDENT_ROLE)
                    self._log_residency("before_student_backward")
                    (result["loss"] / grad_accum_iters).backward()
                    self._log_residency("after_student_backward")
            if offload is not None:
                logger.info(
                    "[h3-a100][activation-offload] component=student stats={}",
                    offload.stats,
                )
            if self.trajectory_recorder is not None:
                self.trajectory_recorder.note_loss("student", result["dmd"])
            running_dmd += result["dmd"].item() / grad_accum_iters

        sync_sequence_parallel_gradients(self.trainable_params)
        torch.nn.utils.clip_grad_norm_(self.trainable_params, self.max_grad_norm)
        if self.trajectory_recorder is not None:
            self.trajectory_recorder.capture_gradient("student", self.trainable_params)
        self._log_residency("before_student_optimizer")
        self.optimizer.step()
        if self.trajectory_recorder is not None:
            self._trajectory_student_optimizer_updates += 1
        self.lr_scheduler.step()
        self._log_residency("after_student_optimizer")
        self.optimizer.zero_grad(set_to_none=True)
        return running_dmd

    def _fake_steps(self, samples: Iterator, grad_accum_iters: int) -> float:
        running_fake = 0.0
        if self.reorder_critic_rollouts:
            with self._nvtx("h3/critic_prepare_5xG"):
                groups = [
                    self._prepare_one_fake_group(samples, grad_accum_iters, fake_index)
                    for fake_index in range(self.fake_update_ratio)
                ]
            for group in groups:
                with self._nvtx("h3/critic_update_F"):
                    running_fake += self._apply_one_fake_group(group) / self.fake_update_ratio
            return running_fake

        for fake_index in range(self.fake_update_ratio):
            group = self._prepare_one_fake_group(samples, grad_accum_iters, fake_index)
            running_fake += self._apply_one_fake_group(group) / self.fake_update_ratio
        return running_fake

    def _prepare_one_fake_group(
        self,
        samples: Iterator,
        grad_accum_iters: int,
        fake_index: int,
    ):
        group = []
        for _ in range(grad_accum_iters):
            sample = next(samples)
            if self.matched_compute_enabled:
                self.matched_cycle_census.note_sample(f"fake_{fake_index}", sample)
            conditions = self._encode_conditions(sample)
            latent_shape = self._latent_shape(sample)
            group.append(self.prepare_fake_update(latent_shape, conditions))
            self._log_residency(f"after_fake_{fake_index}_rollout_prepare")
        return group

    def _apply_one_fake_group(self, group: list[PreparedFakeUpdate]) -> float:
        self.fake_optimizer.zero_grad(set_to_none=True)
        fake_loss_value = 0.0
        for micro_idx, item in enumerate(group):
            self._set_shared_gradient_sync(micro_idx == len(group) - 1)
            with self._activation_offload_scope(f"fake_{micro_idx}") as offload:
                self._log_residency(f"before_fake_{micro_idx}_grad_forward")
                loss_fake = self.fake_loss(item)
                self._log_residency(f"after_fake_{micro_idx}_grad_forward")
                if offload is not None:
                    offload.begin_backward()
                self.shared_model.transformer.train()
                with self.shared_model.role_scope(FAKE_ADAPTER), self.shared_model.adaln_scope(
                    item.cache_key, persistent=False
                ):
                    if self.matched_compute_enabled:
                        self.matched_cycle_census.note_backward(FAKE_ROLE)
                    self._log_residency(f"before_fake_{micro_idx}_backward")
                    (loss_fake / len(group)).backward()
                    self._log_residency(f"after_fake_{micro_idx}_backward")
            if offload is not None:
                logger.info(
                    "[h3-a100][activation-offload] component=fake_{} stats={}",
                    micro_idx,
                    offload.stats,
                )
            self.shared_model.drop_adaln_key(item.cache_key)
            if self.trajectory_recorder is not None:
                self.trajectory_recorder.note_loss("fake", loss_fake)
            fake_loss_value += loss_fake.item() / len(group)

        sync_sequence_parallel_gradients(self.fake_trainable_params)
        torch.nn.utils.clip_grad_norm_(self.fake_trainable_params, self.max_grad_norm)
        if self.trajectory_recorder is not None:
            self.trajectory_recorder.capture_gradient("fake", self.fake_trainable_params)
        self._log_residency("before_fake_optimizer")
        self.fake_optimizer.step()
        if self.trajectory_recorder is not None:
            self._trajectory_fake_optimizer_updates += 1
        self.fake_lr_scheduler.step()
        self._log_residency("after_fake_optimizer")
        self.fake_optimizer.zero_grad(set_to_none=True)
        return fake_loss_value

    def _begin_boundary_offload_cycle(self) -> None:
        registration = getattr(self, "boundary_offload_registration", None)
        if registration is None:
            self._boundary_offload_cycle_start = None
            return
        self._boundary_offload_cycle_start = dict(registration.stats)

    def _validate_boundary_offload_cycle(self, current_iter: int) -> None:
        registration = getattr(self, "boundary_offload_registration", None)
        if registration is None:
            return
        start = self._boundary_offload_cycle_start
        if start is None:
            raise RuntimeError("checkpoint-boundary offload cycle baseline is missing")
        stats = registration.stats

        def delta(key: str) -> int:
            return int(stats[key]) - int(start.get(key, 0))

        # Matched H3 has one Student grad DiT and five Fake grad DiTs.  Native
        # per-block checkpointing therefore produces 6 * 50 checkpoint-input
        # saves.  Recompute itself is outside the save hook.
        expected = {
            "grad_transformer_forward_count": 6,
            "student_grad_forward_count": 1,
            "fake_grad_forward_count": 5,
            "other_grad_forward_count": 0,
            "grad_checkpoint_call_count": 300,
            "cpu_copy_count": 300,
        }
        observed = {key: delta(key) for key in expected}
        errors = [
            f"{key}={observed[key]} expected={value}"
            for key, value in expected.items()
            if observed[key] != value
        ]
        if errors:
            raise RuntimeError(
                "checkpoint-boundary CPU staging contract failed at outer iteration "
                f"{current_iter}: {'; '.join(errors)}"
            )
        offloaded = delta("offloaded_storage_bytes")
        logger.info(
            "[h3-a100][boundary-offload] iter={} rank={} policy={} observed={} "
            "offloaded_gib={:.2f}",
            current_iter,
            get_rank(),
            registration.receipt()["policy"],
            observed,
            offloaded / 1024**3,
        )

    def _validate_matched_cycle(self, current_iter: int) -> dict:
        local_errors = self.matched_cycle_census.validate_local()
        snapshot = self.matched_cycle_census.snapshot()
        world_size = get_world_size()
        snapshots = [None for _ in range(world_size)]
        if dist.is_available() and dist.is_initialized():
            dist.all_gather_object(snapshots, snapshot)
        else:
            snapshots = [snapshot]
        global_errors = validate_global_snapshots(
            snapshots,
            expected_world_size=self.matched_expected_world_size,
            require_unique_samples=self.matched_require_unique_samples,
        )
        errors = local_errors + global_errors
        if errors:
            raise RuntimeError(
                "Matched MiniMax compute contract failed at outer iteration "
                f"{current_iter}: {'; '.join(errors[:12])}"
            )
        logger.info(
            "[h3-a100][matched] iter={} rank={} forwards={} grad_forwards={} "
            "backwards={} samples={} world={}",
            current_iter,
            get_rank(),
            snapshot["forward_counts"],
            snapshot["grad_forward_counts"],
            snapshot["backward_counts"],
            len(snapshot["samples"]),
            world_size,
        )
        return snapshot

    def _set_shared_gradient_sync(self, enabled):
        set_parallel_gradient_sync(self.shared_model, enabled)

    @contextlib.contextmanager
    def _nvtx(self, name: str):
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_push(name)
            try:
                yield
            finally:
                torch.cuda.nvtx.range_pop()
        else:
            yield

    def _release_full_cpu_checkpoint_storage(self) -> None:
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError):
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _log_cuda_memory(self, stage: str) -> None:
        if not torch.cuda.is_available():
            return
        logger.info(
            "[h3-a100][memory] stage={} allocated_gib={:.2f} reserved_gib={:.2f} peak_gib={:.2f}",
            stage,
            torch.cuda.memory_allocated() / 1024**3,
            torch.cuda.memory_reserved() / 1024**3,
            torch.cuda.max_memory_allocated() / 1024**3,
        )

    def _log_residency(self, stage: str) -> None:
        """Emit bounded GPU residency attribution for OOM diagnosis only.

        This is disabled by default and intentionally reads allocator/model
        metadata without synchronizing or changing the training graph.  It is
        not part of the timing contract.
        """
        if os.environ.get("H3_MEMORY_ATTRIBUTION", "0").lower() not in {"1", "true", "yes", "on"}:
            return
        if not torch.cuda.is_available():
            return
        stats = torch.cuda.memory_stats()
        controller = None
        try:
            controller = self.shared_model.adaln_cache().stats()
        except Exception:
            pass

        def tensor_bytes(value):
            if not torch.is_tensor(value):
                return 0
            try:
                return int(value.numel() * value.element_size())
            except Exception:
                return 0

        def optimizer_bytes(optimizer):
            total = 0
            for state in optimizer.state.values():
                for value in state.values():
                    total += tensor_bytes(value)
            return total

        params = {"all": 0, "lora_student": 0, "lora_fake": 0, "frozen": 0}
        try:
            for name, parameter in self.shared_model.denoiser_module().named_parameters():
                size = tensor_bytes(parameter)
                params["all"] += size
                if "lora_" in name and "student" in name:
                    params["lora_student"] += size
                elif "lora_" in name and "fake" in name:
                    params["lora_fake"] += size
                elif not parameter.requires_grad:
                    params["frozen"] += size
        except Exception:
            pass

        boundary = {}
        registration = getattr(self, "boundary_offload_registration", None)
        if registration is not None:
            for key in (
                "cpu_copy_count",
                "offloaded_logical_bytes",
                "offloaded_storage_bytes",
                "pack_count",
                "unpack_count",
            ):
                boundary[key] = int(registration.stats.get(key, 0))

        free, total = torch.cuda.mem_get_info()
        logger.info(
            "[h3-a100][residency] stage={} allocated={} reserved={} active={} "
            "inactive_split={} free={} total={} params={} opt_student={} opt_fake={} "
            "adaln_cache={} boundary={}",
            stage,
            int(torch.cuda.memory_allocated()),
            int(torch.cuda.memory_reserved()),
            int(stats.get("active_bytes.all.current", 0)),
            int(stats.get("inactive_split_bytes.all.current", 0)),
            int(free),
            int(total),
            params,
            optimizer_bytes(getattr(self, "optimizer", None)) if getattr(self, "optimizer", None) else 0,
            optimizer_bytes(getattr(self, "fake_optimizer", None)) if getattr(self, "fake_optimizer", None) else 0,
            0 if controller is None else int(controller.bytes),
            boundary,
        )
