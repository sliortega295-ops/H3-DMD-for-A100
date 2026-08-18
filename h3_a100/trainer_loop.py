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
        samples = self._iter_train_samples()
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

        while current_iter < self.max_train_iters:
            if self.matched_compute_enabled:
                self.matched_cycle_census.reset()
            with self._nvtx("h3/student_step"):
                running_dmd = self._student_step(samples, grad_accum_iters, current_iter)
            with self._nvtx("h3/critic_phase"):
                running_fake = self._fake_steps(samples, grad_accum_iters)
            if self.matched_compute_enabled:
                self._validate_matched_cycle(current_iter)

            current_iter += 1
            display_dmd = reduce_mean(running_dmd)
            display_fake = reduce_mean(running_fake)
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
            result = self.forward_student_loss(latent_shape, conditions, current_iter=current_iter)
            self.shared_model.transformer.train()
            with self.shared_model.role_scope(STUDENT_ADAPTER), self.shared_model.adaln_scope(
                result["backward_key"], persistent=True
            ):
                if self.matched_compute_enabled:
                    self.matched_cycle_census.note_backward(STUDENT_ROLE)
                (result["loss"] / grad_accum_iters).backward()
            running_dmd += result["dmd"].item() / grad_accum_iters

        sync_sequence_parallel_gradients(self.trainable_params)
        torch.nn.utils.clip_grad_norm_(self.trainable_params, self.max_grad_norm)
        self.optimizer.step()
        self.lr_scheduler.step()
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
        return group

    def _apply_one_fake_group(self, group: list[PreparedFakeUpdate]) -> float:
        self.fake_optimizer.zero_grad(set_to_none=True)
        fake_loss_value = 0.0
        for micro_idx, item in enumerate(group):
            self._set_shared_gradient_sync(micro_idx == len(group) - 1)
            loss_fake = self.fake_loss(item)
            self.shared_model.transformer.train()
            with self.shared_model.role_scope(FAKE_ADAPTER), self.shared_model.adaln_scope(
                item.cache_key, persistent=False
            ):
                if self.matched_compute_enabled:
                    self.matched_cycle_census.note_backward(FAKE_ROLE)
                (loss_fake / len(group)).backward()
            self.shared_model.drop_adaln_key(item.cache_key)
            fake_loss_value += loss_fake.item() / len(group)

        sync_sequence_parallel_gradients(self.fake_trainable_params)
        torch.nn.utils.clip_grad_norm_(self.fake_trainable_params, self.max_grad_norm)
        self.fake_optimizer.step()
        self.fake_lr_scheduler.step()
        self.fake_optimizer.zero_grad(set_to_none=True)
        return fake_loss_value

    def _validate_matched_cycle(self, current_iter: int) -> None:
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
