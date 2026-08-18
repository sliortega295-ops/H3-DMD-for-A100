"""H3 shared-backbone rollout, score, and AdaLN-cache runtime methods."""

from __future__ import annotations

import dataclasses
from typing import Any

import torch

from lightx2v_train.model_zoo.native.minimax_h3 import build_row_timesteps
from lightx2v_train.trainers.dmd.math import dmd_loss_pair, weighted_mse_pair

from .distributed import broadcast_shard_tensor, shard_source
from .model import BASE_ROLE, FAKE_ADAPTER, STUDENT_ADAPTER


@dataclasses.dataclass
class PreparedFakeUpdate:
    latent_shape: dict[str, Any]
    condition: dict[str, torch.Tensor]
    generated: tuple[torch.Tensor, torch.Tensor]
    noises: tuple[torch.Tensor, torch.Tensor]
    renoised: tuple[torch.Tensor, torch.Tensor]
    sigmas: tuple[torch.Tensor, torch.Tensor]
    cache_key: tuple[str, int]


class H3A100RuntimeMixin:
    """Methods that execute Student/Fake/Teacher forwards on one module."""

    # ------------------------------------------------------------------
    # HSDP input synchronization
    # ------------------------------------------------------------------
    def sample_initial_latents(self, latent_shape):
        video = torch.empty(
            latent_shape["video_tokens"], device=self.shared_model.device, dtype=self.latent_dtype
        )
        audio = torch.empty(
            latent_shape["audio_tokens"], device=self.shared_model.device, dtype=self.latent_dtype
        )
        if shard_source():
            video.normal_()
            audio.normal_()
        broadcast_shard_tensor(video)
        broadcast_shard_tensor(audio)
        return video, audio

    def _sample_synced_int(self, low, high):
        value = torch.empty((), device=self.shared_model.device, dtype=torch.int64)
        if shard_source():
            value.random_(int(low), int(high))
        broadcast_shard_tensor(value)
        return int(value.item())

    def _sample_renoise_sigmas(self):
        low = float(self.dmd_config.get("renoise_sigma_min", 0.02))
        high = float(self.dmd_config.get("renoise_sigma_max", 0.98))
        if not 0.0 <= low < high <= 1.0:
            raise ValueError(
                f"H3 renoise sigma range must satisfy 0 <= min < max <= 1, got [{low}, {high}]"
            )
        base = torch.empty((), device=self.shared_model.device, dtype=torch.float32)
        if shard_source():
            base.uniform_(low, high)
        broadcast_shard_tensor(base)
        video = self.video_shift * base / (1.0 + (self.video_shift - 1.0) * base)
        audio = self.audio_shift * base / (1.0 + (self.audio_shift - 1.0) * base)
        return video, audio

    def _randn_like_synced(self, tensor):
        noise = torch.empty_like(tensor, dtype=torch.float32)
        if shard_source():
            noise.normal_()
        broadcast_shard_tensor(noise)
        return noise

    # ------------------------------------------------------------------
    # Role switching and AdaLN precomputation
    # ------------------------------------------------------------------
    def _predict_velocity_role(self, role, latents, sigmas, condition, latent_shape):
        with self.shared_model.role_scope(role):
            return super()._predict_velocity(
                self.shared_model, latents, sigmas, condition, latent_shape
            )

    def _adaln_timesteps(self, condition, latent_shape, sigmas):
        layout = self._layout(condition, latent_shape)
        timesteps, _ = build_row_timesteps(layout, sigmas[0], sigmas[1])
        return timesteps.to(self.shared_model.device)

    def run_back_simulation(self, condition, latent_shape, end_step_idx, grad_enabled, xt=None):
        if xt is None:
            xt = self.sample_initial_latents(latent_shape)
        self.shared_model.transformer.train()
        x0 = None
        backward_key = None
        for step_idx in range(end_step_idx + 1):
            sigmas = (
                self.video_sigmas_cpu[step_idx].to(self.shared_model.device),
                self.audio_sigmas_cpu[step_idx].to(self.shared_model.device),
            )
            key = ("rollout", int(step_idx))
            timesteps = self._adaln_timesteps(condition, latent_shape, sigmas)
            self.shared_model.precompute_adaln(key, timesteps, persistent=True)
            context = torch.enable_grad if grad_enabled and step_idx == end_step_idx else torch.no_grad
            with context(), self.shared_model.adaln_scope(key, persistent=True):
                velocity = self._predict_velocity_role(
                    STUDENT_ADAPTER, xt, sigmas, condition, latent_shape
                )
                x0 = (
                    xt[0] + sigmas[0] * velocity[0],
                    xt[1] + sigmas[1] * velocity[1],
                )
            if grad_enabled and step_idx == end_step_idx:
                backward_key = key
            next_sigmas = (
                self.video_sigmas_cpu[step_idx + 1].to(self.shared_model.device),
                self.audio_sigmas_cpu[step_idx + 1].to(self.shared_model.device),
            )
            xt = (
                (xt[0].float() + (sigmas[0] - next_sigmas[0]) * velocity[0].float()).to(
                    self.latent_dtype
                ),
                (xt[1].float() + (sigmas[1] - next_sigmas[1]) * velocity[1].float()).to(
                    self.latent_dtype
                ),
            )
        if x0 is None:
            raise RuntimeError("H3 rollout produced no x0 prediction")
        return (x0[0].to(self.latent_dtype), x0[1].to(self.latent_dtype)), backward_key

    def forward_student_loss(self, latent_shape, conditions, current_iter=None):
        del current_iter
        condition, _ = conditions
        end_step_idx = self._sample_synced_int(0, self.num_inference_steps)
        generated, backward_key = self.run_back_simulation(
            condition, latent_shape, end_step_idx, grad_enabled=True
        )
        sigmas = self._sample_renoise_sigmas()
        noises = (
            self._randn_like_synced(generated[0]),
            self._randn_like_synced(generated[1]),
        )
        renoised = self._add_noise(generated, noises, sigmas)

        score_key = self._next_score_key()
        timesteps = self._adaln_timesteps(condition, latent_shape, sigmas)
        self.shared_model.precompute_adaln(score_key, timesteps, persistent=False)
        self.shared_model.transformer.eval()
        with torch.no_grad(), self.shared_model.adaln_scope(score_key, persistent=False):
            velocity_fake = self._predict_velocity_role(
                FAKE_ADAPTER, renoised, sigmas, condition, latent_shape
            )
            velocity_teacher = self._predict_velocity_role(
                BASE_ROLE, renoised, sigmas, condition, latent_shape
            )
            x0_fake = (
                renoised[0] + sigmas[0] * velocity_fake[0],
                renoised[1] + sigmas[1] * velocity_fake[1],
            )
            x0_teacher = (
                renoised[0] + sigmas[0] * velocity_teacher[0],
                renoised[1] + sigmas[1] * velocity_teacher[1],
            )
        self.shared_model.drop_adaln_key(score_key)
        loss = dmd_loss_pair(
            generated,
            x0_fake,
            x0_teacher,
            self.video_loss_weight,
            self.audio_loss_weight,
        )
        if backward_key is None:
            raise RuntimeError("Student DMD loss has no rollout backward cache key")
        return {"loss": loss, "dmd": loss.detach(), "backward_key": backward_key}

    @torch.no_grad()
    def prepare_fake_update(self, latent_shape, conditions) -> PreparedFakeUpdate:
        condition, _ = conditions
        end_step_idx = self._sample_synced_int(0, self.num_inference_steps)
        generated, _ = self.run_back_simulation(
            condition, latent_shape, end_step_idx, grad_enabled=False
        )
        generated = (generated[0].detach(), generated[1].detach())
        sigmas = self._sample_renoise_sigmas()
        noises = (
            self._randn_like_synced(generated[0]),
            self._randn_like_synced(generated[1]),
        )
        renoised = self._add_noise(generated, noises, sigmas)
        return PreparedFakeUpdate(
            latent_shape=latent_shape,
            condition=condition,
            generated=generated,
            noises=noises,
            renoised=renoised,
            sigmas=sigmas,
            cache_key=self._next_score_key(),
        )

    def fake_loss(self, item: PreparedFakeUpdate):
        timesteps = self._adaln_timesteps(item.condition, item.latent_shape, item.sigmas)
        self.shared_model.precompute_adaln(item.cache_key, timesteps, persistent=False)
        self.shared_model.transformer.train()
        with self.shared_model.adaln_scope(item.cache_key, persistent=False):
            velocity_fake = self._predict_velocity_role(
                FAKE_ADAPTER,
                item.renoised,
                item.sigmas,
                item.condition,
                item.latent_shape,
            )
            target = (
                item.generated[0].float() - item.noises[0],
                item.generated[1].float() - item.noises[1],
            )
            loss = weighted_mse_pair(
                velocity_fake,
                target,
                self.video_loss_weight,
                self.audio_loss_weight,
            )
        return loss

    def _next_score_key(self):
        self._score_serial += 1
        return ("score", self._score_serial)
