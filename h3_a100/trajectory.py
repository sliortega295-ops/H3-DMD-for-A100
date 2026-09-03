"""Low-overhead, append-only receipts for matched 50-cycle trajectories.

The trajectory mode is deliberately diagnostic: it does not participate in
formal timing and is disabled unless ``H3_TRAJECTORY_MODE=coarse_v1``.  The
operation-keyed sigma sampler is shared by Exact and Grid-1000 runs so that the
Grid arm snaps the *same* continuous draw instead of consuming a different RNG
stream.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import struct
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

COARSE_SCHEMA = "h3-trajectory-coarse/v1"
MODE_NAME = "coarse_v1"
SIGMA_OPERATIONS = ("student", "fake_0", "fake_1", "fake_2", "fake_3", "fake_4")
DEFAULT_ANCHORS = (1, 10, 25, 50)


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def _shift_sigma(value: float, shift: float) -> float:
    value = _f32(value)
    shift = _f32(shift)
    return _f32(_f32(shift * value) / _f32(1.0 + _f32((shift - 1.0) * value)))


@dataclass(frozen=True)
class SigmaSample:
    operation_key: str
    continuous_base: float
    actual_base: float
    video_sigma: float
    audio_sigma: float
    grid_index: int | None
    snap_abs_error: float
    low: float
    high: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class OperationKeyedSigmaSampler:
    """Generate rank/cycle/operation-qualified continuous sigma draws.

    This intentionally does not touch Python, NumPy, CPU torch, or CUDA torch
    RNG state.  Therefore Exact and Grid runs consume identical model/noise RNG
    streams while their only registered difference is continuous vs snapped
    sigma.
    """

    def __init__(self, *, seed: int, rank: int, variant: str, grid_size: int = 1000):
        if variant not in {"exact", "grid1000"}:
            raise ValueError(f"trajectory sigma variant must be exact or grid1000, got {variant!r}")
        if grid_size < 2:
            raise ValueError("grid_size must be at least 2")
        self.seed = int(seed)
        self.rank = int(rank)
        self.variant = variant
        self.grid_size = int(grid_size)
        self._grids: dict[tuple[float, float], torch.Tensor] = {}

    @staticmethod
    def operation_name(slot: int) -> str:
        try:
            return SIGMA_OPERATIONS[int(slot)]
        except (IndexError, ValueError) as exc:
            raise RuntimeError(f"H3 trajectory expected six renoise sigma calls, got slot {slot}") from exc

    def sample(self, *, cycle: int, slot: int, low: float, high: float) -> SigmaSample:
        low = _f32(low)
        high = _f32(high)
        if not 0.0 <= low < high <= 1.0:
            raise ValueError(f"invalid renoise sigma interval [{low}, {high}]")
        operation = self.operation_name(slot)
        operation_key = f"rank={self.rank}/cycle={int(cycle)}/{operation}/renoise_sigma"
        payload = f"{COARSE_SCHEMA}|seed={self.seed}|rank={self.rank}|{operation_key}".encode()
        integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        unit = integer / float(1 << 64)
        continuous = _f32(low + (high - low) * unit)
        grid_index = None
        actual = continuous
        if self.variant == "grid1000":
            position = (continuous - low) / (high - low) * (self.grid_size - 1)
            grid_index = min(self.grid_size - 1, max(0, int(math.floor(position + 0.5))))
            # Mirror torch.linspace(..., dtype=float32) used by the Grid runtime.
            # Cache this tiny CPU vector so a 50-cycle trajectory does not
            # rebuild it for every operation.
            grid_key = (low, high)
            grid = self._grids.get(grid_key)
            if grid is None:
                grid = torch.linspace(low, high, self.grid_size, dtype=torch.float32)
                self._grids[grid_key] = grid
            actual = float(grid[grid_index])
        return SigmaSample(
            operation_key=operation_key,
            continuous_base=continuous,
            actual_base=_f32(actual),
            video_sigma=_shift_sigma(actual, 6.0),
            audio_sigma=_shift_sigma(actual, 3.0),
            grid_index=grid_index,
            snap_abs_error=abs(_f32(actual) - continuous),
            low=low,
            high=high,
        )


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_head() -> str:
    root = Path(__file__).resolve().parents[1]
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNRESOLVED"


def _environment_path(name: str) -> Path | None:
    raw = os.environ.get(name, "").strip()
    return Path(raw).resolve() if raw else None


def runtime_identity(*, variant: str) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    config_path = _environment_path("H3_TRAJECTORY_CONFIG_PATH")
    model_root = _environment_path("MINIMAX_H3_MODEL_PATH")
    prompt_root = _environment_path("H3_PROMPT_CACHE")
    grid_manifest = _environment_path("H3_ADALN_GRID_MANIFEST")
    upstream = root / "UPSTREAM_COMMIT"

    def package_version(name: str) -> str:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return "UNRESOLVED"

    return {
        "source_head": _repo_head(),
        "source_expected_head": os.environ.get("H3_EXPECTED_HEAD", "NOT_PROVIDED"),
        "upstream_lightx2v": upstream.read_text().strip() if upstream.is_file() else "UNRESOLVED",
        "variant": variant,
        "python": os.sys.version.split()[0],
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "diffusers": package_version("diffusers"),
        "peft": package_version("peft"),
        "config_path": str(config_path) if config_path else "NOT_PROVIDED",
        "config_sha256": _sha256(config_path) if config_path else None,
        "model_root": str(model_root) if model_root else "NOT_PROVIDED",
        "model_config_sha256": (
            _sha256(model_root / "transformer" / "config.json") if model_root else None
        ),
        "prompt_cache": str(prompt_root) if prompt_root else "NOT_PROVIDED",
        "prompt_metadata_sha256": (
            _sha256(prompt_root / "metadata.jsonl") if prompt_root else None
        ),
        "grid_manifest": str(grid_manifest) if grid_manifest else None,
        "grid_manifest_sha256": _sha256(grid_manifest) if grid_manifest else None,
        "attention_backend": os.environ.get("H3_ATTN_BACKEND", "_flash_3_hub"),
        "activation_policy": os.environ.get("H3_ACTIVATION_POLICY", "config_default"),
        "checkpoint_segment": int(os.environ.get("H3_ACTIVATION_CHECKPOINT_SEGMENT", "1")),
        "h3_environment": {
            key: value
            for key, value in sorted(os.environ.items())
            if key.startswith("H3_") and key not in {"H3_TRAJECTORY_DIR"}
        },
    }


def _local_tensor(value: torch.Tensor) -> torch.Tensor:
    value = value.detach()
    to_local = getattr(value, "to_local", None)
    if callable(to_local):
        value = to_local()
    return value


def _sample_indices(numel: int, count: int) -> list[int]:
    if numel <= 0 or count <= 0:
        return []
    count = min(numel, count)
    if count == 1:
        return [0]
    return sorted({round(index * (numel - 1) / (count - 1)) for index in range(count)})


def _summarize_values(coordinates: list[str], values: list[float]) -> dict[str, Any]:
    packed = bytearray()
    for coordinate, value in zip(coordinates, values, strict=True):
        packed.extend(coordinate.encode())
        packed.append(0)
        packed.extend(struct.pack("<f", float(value)))
    finite = [float(value) for value in values]
    l2 = math.sqrt(sum(value * value for value in finite))
    return {
        "sample_count": len(finite),
        "l2": l2,
        "mean": sum(finite) / len(finite) if finite else 0.0,
        "max_abs": max((abs(value) for value in finite), default=0.0),
        "sha256": hashlib.sha256(packed).hexdigest(),
        "coordinates": coordinates,
        "values": finite,
    }


def _sample_named_tensors(
    named_tensors: Iterable[tuple[str, torch.Tensor]],
    *,
    samples_per_tensor: int,
    max_tensors: int,
) -> tuple[list[str], list[float]]:
    coordinates: list[str] = []
    values: list[float] = []
    ordered = sorted(named_tensors, key=lambda item: item[0])
    if len(ordered) > max_tensors:
        selected_indices = _sample_indices(len(ordered), max_tensors)
        ordered = [ordered[index] for index in selected_indices]
    for name, tensor in ordered:
        local = _local_tensor(tensor)
        if local.numel() == 0:
            continue
        flat = local.reshape(-1)
        indices = _sample_indices(flat.numel(), samples_per_tensor)
        selected = flat[torch.tensor(indices, device=flat.device, dtype=torch.long)]
        selected_values = selected.float().cpu().tolist()
        coordinates.extend(f"{name}[{index}]" for index in indices)
        values.extend(float(value) for value in selected_values)
    return coordinates, values


class TrajectoryRecorder:
    """Append-only per-rank trajectory and sampled state evidence."""

    def __init__(
        self,
        *,
        output_root: Path,
        rank: int,
        world_size: int,
        seed: int,
        variant: str,
        expected_cycles: int = 50,
        anchors: Sequence[int] = DEFAULT_ANCHORS,
        samples_per_tensor: int = 4,
        max_tensors_per_role: int = 32,
        identity: Mapping[str, Any] | None = None,
    ):
        self.output_root = Path(output_root)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.variant = variant
        self.expected_cycles = int(expected_cycles)
        self.anchors = tuple(sorted({int(value) for value in anchors}))
        self.samples_per_tensor = int(samples_per_tensor)
        self.max_tensors_per_role = int(max_tensors_per_role)
        if self.expected_cycles < 1:
            raise ValueError("trajectory expected_cycles must be positive")
        if not self.anchors or any(value < 1 or value > self.expected_cycles for value in self.anchors):
            raise ValueError("trajectory anchors must fall inside expected cycles")
        if self.samples_per_tensor < 1:
            raise ValueError("samples_per_tensor must be positive")
        if self.max_tensors_per_role < 1:
            raise ValueError("max_tensors_per_role must be positive")
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.path = self.output_root / f"rank_{self.rank:03d}.trajectory.jsonl"
        try:
            self.path.open("x", encoding="utf-8").close()
        except FileExistsError as exc:
            raise FileExistsError(
                f"Refusing to overwrite existing trajectory evidence: {self.path}"
            ) from exc
        self.manifest_path = self.output_root / "trajectory_manifest.json"
        if self.rank == 0 and self.manifest_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing trajectory manifest: {self.manifest_path}"
            )
        self.identity = dict(identity or runtime_identity(variant=variant))
        self.sampler = OperationKeyedSigmaSampler(seed=seed, rank=rank, variant=variant)
        self._cycle: int | None = None
        self._sigma_samples: list[SigmaSample] = []
        self._losses: dict[str, list[torch.Tensor]] = {"student": [], "fake": []}
        self._gradients: dict[str, list[dict[str, Any]]] = {"student": [], "fake": []}
        self._named_parameter_by_id: dict[int, str] = {}
        self._initial_samples: dict[str, tuple[list[str], list[float]]] = {}
        self._records_written = 0

    @classmethod
    def from_environment(cls, *, rank: int, world_size: int, seed: int, variant: str):
        mode = os.environ.get("H3_TRAJECTORY_MODE", "off").strip().lower()
        if mode in {"", "0", "off", "false", "none"}:
            return None
        if mode != MODE_NAME:
            raise ValueError(f"H3_TRAJECTORY_MODE must be {MODE_NAME!r} or off, got {mode!r}")
        requested_variant = os.environ.get("H3_TRAJECTORY_VARIANT", variant)
        if requested_variant != variant:
            raise RuntimeError(
                f"trajectory launcher requested {requested_variant!r}, runtime is {variant!r}"
            )
        output = os.environ.get("H3_TRAJECTORY_DIR", "")
        if not output:
            raise RuntimeError("H3_TRAJECTORY_DIR is required when trajectory mode is enabled")
        expected_world_size = int(os.environ.get("H3_TRAJECTORY_EXPECTED_WORLD_SIZE", "16"))
        if int(world_size) != expected_world_size:
            raise RuntimeError(
                f"trajectory requires world_size={expected_world_size}, got {world_size}"
            )
        expected_cycles = int(os.environ.get("H3_TRAJECTORY_EXPECTED_CYCLES", "50"))
        anchors = tuple(
            int(value)
            for value in os.environ.get("H3_TRAJECTORY_ANCHORS", "1,10,25,50").split(",")
            if value.strip()
        )
        samples_per_tensor = int(os.environ.get("H3_TRAJECTORY_SAMPLES_PER_TENSOR", "4"))
        max_tensors_per_role = int(
            os.environ.get("H3_TRAJECTORY_MAX_TENSORS_PER_ROLE", "32")
        )
        return cls(
            output_root=Path(output),
            rank=rank,
            world_size=world_size,
            seed=seed,
            variant=variant,
            expected_cycles=expected_cycles,
            anchors=anchors,
            samples_per_tensor=samples_per_tensor,
            max_tensors_per_role=max_tensors_per_role,
        )

    def _append(self, payload: Mapping[str, Any]) -> None:
        row = {"schema": COARSE_SCHEMA, **payload}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        self._records_written += 1

    def _role_named(self, parameters: Iterable[torch.Tensor]) -> list[tuple[str, torch.Tensor]]:
        result = []
        for parameter in parameters:
            name = self._named_parameter_by_id.get(id(parameter))
            if name is None:
                raise RuntimeError("trajectory state probe cannot map an optimizer parameter to a name")
            result.append((name, parameter))
        return result

    def start_run(
        self,
        *,
        current_iter: int,
        max_train_iters: int,
        named_parameters: Mapping[str, torch.Tensor] | Iterable[tuple[str, torch.Tensor]],
        student_parameters: Sequence[torch.Tensor],
        fake_parameters: Sequence[torch.Tensor],
    ) -> None:
        if int(current_iter) != 0:
            raise RuntimeError("50-cycle trajectory evidence requires a fresh restart at iteration 0")
        if int(max_train_iters) != self.expected_cycles:
            raise RuntimeError(
                f"trajectory requires H3_MAX_ITERS={self.expected_cycles}, got {max_train_iters}"
            )
        items = list(named_parameters.items() if isinstance(named_parameters, Mapping) else named_parameters)
        self._named_parameter_by_id = {id(parameter): name for name, parameter in items}
        for role, parameters in (("student", student_parameters), ("fake", fake_parameters)):
            coords, values = _sample_named_tensors(
                self._role_named(parameters),
                samples_per_tensor=self.samples_per_tensor,
                max_tensors=self.max_tensors_per_role,
            )
            self._initial_samples[role] = (coords, values)
        start = {
            "record_type": "run_start",
            "rank": self.rank,
            "world_size": self.world_size,
            "variant": self.variant,
            "seed": self.seed,
            "expected_cycles": self.expected_cycles,
            "anchors": list(self.anchors),
            "state_sketch": {
                "max_tensors_per_role": self.max_tensors_per_role,
                "samples_per_tensor": self.samples_per_tensor,
            },
            "identity": self.identity,
            "initial_state": {
                role: _summarize_values(*values) for role, values in self._initial_samples.items()
            },
        }
        self._append(start)
        if self.rank == 0:
            with self.manifest_path.open("x", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {"schema": COARSE_SCHEMA, **start},
                        indent=2,
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n"
                )

    def begin_cycle(self, cycle: int) -> None:
        if self._cycle is not None:
            raise RuntimeError(f"trajectory cycle {self._cycle} was not finished")
        self._cycle = int(cycle)
        self._sigma_samples = []
        self._losses = {"student": [], "fake": []}
        self._gradients = {"student": [], "fake": []}

    def sample_renoise_sigmas(
        self,
        *,
        low: float,
        high: float,
        video_shift: float,
        audio_shift: float,
        device: torch.device | str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._cycle is None:
            raise RuntimeError("trajectory sigma sampled outside an active cycle")
        sample = self.sampler.sample(
            cycle=self._cycle,
            slot=len(self._sigma_samples),
            low=low,
            high=high,
        )
        # Recompute shifts using the runtime's configured values and float32
        # operations. The dataclass keeps them for exact audit receipts.
        sample = SigmaSample(
            **{
                **sample.as_dict(),
                "video_sigma": _shift_sigma(sample.actual_base, video_shift),
                "audio_sigma": _shift_sigma(sample.actual_base, audio_shift),
            }
        )
        self._sigma_samples.append(sample)
        base = torch.tensor(sample.actual_base, dtype=torch.float32)
        video = (float(video_shift) * base / (1.0 + (float(video_shift) - 1.0) * base)).to(device)
        audio = (float(audio_shift) * base / (1.0 + (float(audio_shift) - 1.0) * base)).to(device)
        return video, audio

    def note_loss(self, role: str, value: torch.Tensor) -> None:
        if role not in self._losses:
            raise ValueError(f"unknown H3 trajectory loss role {role!r}")
        self._losses[role].append(value.detach())

    def capture_gradient(self, role: str, parameters: Sequence[torch.Tensor]) -> None:
        if self._cycle is None or self._cycle + 1 not in self.anchors:
            return
        named = [
            (name, parameter.grad)
            for name, parameter in self._role_named(parameters)
            if parameter.grad is not None
        ]
        coords, values = _sample_named_tensors(
            named,
            samples_per_tensor=self.samples_per_tensor,
            max_tensors=self.max_tensors_per_role,
        )
        self._gradients[role].append(_summarize_values(coords, values))

    def _state_anchor(
        self,
        *,
        role: str,
        parameters: Sequence[torch.Tensor],
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, Any]:
        named = self._role_named(parameters)
        coords, values = _sample_named_tensors(
            named,
            samples_per_tensor=self.samples_per_tensor,
            max_tensors=self.max_tensors_per_role,
        )
        initial_coords, initial_values = self._initial_samples[role]
        if coords != initial_coords:
            raise RuntimeError(f"trajectory {role} parameter sample coordinates changed")
        delta = [value - initial for value, initial in zip(values, initial_values, strict=True)]
        result: dict[str, Any] = {
            "gradient": self._gradients[role][0] if role == "student" else self._gradients[role],
            "parameter_delta": _summarize_values(coords, delta),
        }
        parameter_name = {id(parameter): name for name, parameter in named}
        for key in ("exp_avg", "exp_avg_sq"):
            state_tensors = []
            for parameter, state in optimizer.state.items():
                value = state.get(key)
                if torch.is_tensor(value) and id(parameter) in parameter_name:
                    state_tensors.append((parameter_name[id(parameter)] + f"/{key}", value))
            state_coords, state_values = _sample_named_tensors(
                state_tensors,
                samples_per_tensor=self.samples_per_tensor,
                max_tensors=self.max_tensors_per_role,
            )
            result[f"adam_{key}"] = _summarize_values(state_coords, state_values)
        return result

    @staticmethod
    def _materialize(values: Sequence[torch.Tensor]) -> list[float]:
        return [float(value.float().cpu().item()) for value in values]

    def finish_cycle(
        self,
        *,
        cycle: int,
        world_dmd: float,
        world_fake: float,
        matched_snapshot: Mapping[str, Any],
        student_parameters: Sequence[torch.Tensor],
        fake_parameters: Sequence[torch.Tensor],
        student_optimizer: torch.optim.Optimizer,
        fake_optimizer: torch.optim.Optimizer,
        student_scheduler_steps: int,
        fake_scheduler_steps: int,
        local_dmd: Sequence[torch.Tensor] | None = None,
        local_fake: Sequence[torch.Tensor] | None = None,
    ) -> None:
        if self._cycle != int(cycle):
            raise RuntimeError(f"trajectory active cycle={self._cycle}, finish requested for {cycle}")
        if len(self._sigma_samples) != len(SIGMA_OPERATIONS):
            raise RuntimeError(
                f"trajectory cycle {cycle} saw {len(self._sigma_samples)} renoise sigma calls; expected 6"
            )
        dmd_values = list(local_dmd) if local_dmd is not None else self._losses["student"]
        fake_values = list(local_fake) if local_fake is not None else self._losses["fake"]
        if len(dmd_values) != 1 or len(fake_values) != 5:
            raise RuntimeError(
                f"trajectory cycle {cycle} loss inventory is Student={len(dmd_values)} Fake={len(fake_values)}; expected 1/5"
            )
        anchor = None
        if cycle + 1 in self.anchors:
            if len(self._gradients["student"]) != 1 or len(self._gradients["fake"]) != 5:
                raise RuntimeError(
                    "trajectory anchor gradient inventory is "
                    f"Student={len(self._gradients['student'])} "
                    f"Fake={len(self._gradients['fake'])}; expected 1/5"
                )
            anchor = {
                "student": self._state_anchor(
                    role="student", parameters=student_parameters, optimizer=student_optimizer
                ),
                "fake": self._state_anchor(
                    role="fake", parameters=fake_parameters, optimizer=fake_optimizer
                ),
            }
        self._append(
            {
                "record_type": "cycle",
                "rank": self.rank,
                "world_size": self.world_size,
                "variant": self.variant,
                "cycle": int(cycle) + 1,
                "losses": {
                    "dmd_local": self._materialize(dmd_values),
                    "fake_local": self._materialize(fake_values),
                    "dmd_world_mean": float(world_dmd),
                    "fake_world_mean": float(world_fake),
                },
                "sigmas": [sample.as_dict() for sample in self._sigma_samples],
                "matched_compute": dict(matched_snapshot),
                "versions": {
                    "student_optimizer_updates": int(cycle) + 1,
                    "fake_optimizer_updates": (int(cycle) + 1) * 5,
                    "student_scheduler_steps": int(student_scheduler_steps),
                    "fake_scheduler_steps": int(fake_scheduler_steps),
                    "ema_updates": 0,
                },
                "state_anchor": anchor,
            }
        )
        self._cycle = None

    def finish_run(self) -> None:
        if self._cycle is not None:
            raise RuntimeError(f"cannot finish trajectory with active cycle {self._cycle}")
        # One run_start record plus one record per completed cycle.
        completed = self._records_written - 1
        if completed != self.expected_cycles:
            raise RuntimeError(
                f"trajectory completed cycles={completed}, expected={self.expected_cycles}"
            )
        self._append(
            {
                "record_type": "run_end",
                "rank": self.rank,
                "world_size": self.world_size,
                "variant": self.variant,
                "completed_cycles": completed,
                "status": "COMPLETE",
            }
        )


def _read_cycle_rows(root: Path) -> dict[int, list[dict[str, Any]]]:
    result: dict[int, list[dict[str, Any]]] = {}
    for path in sorted(Path(root).glob("rank_*.trajectory.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        cycles = [row for row in rows if row.get("record_type") == "cycle"]
        if not cycles:
            continue
        rank = int(cycles[0]["rank"])
        result[rank] = cycles
    if not result:
        raise RuntimeError(f"no trajectory cycle receipts found under {root}")
    return result


def _read_start_rows(root: Path) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for path in sorted(Path(root).glob("rank_*.trajectory.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("record_type") == "run_start":
                result[int(row["rank"])] = row
                break
    return result


def _curve(rows: dict[int, list[dict[str, Any]]], key: str, fake_index: int | None = None):
    cycle_count = len(next(iter(rows.values())))
    values = []
    for index in range(cycle_count):
        rank_values = []
        for rank_rows in rows.values():
            losses = rank_rows[index]["losses"]
            if fake_index is None:
                rank_values.extend(float(value) for value in losses[key])
            else:
                rank_values.append(float(losses[key][fake_index]))
        values.append(sum(rank_values) / len(rank_values))
    return values


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return float("nan")
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    denom = math.sqrt(
        sum(value * value for value in left_centered)
        * sum(value * value for value in right_centered)
    )
    if denom == 0:
        return 1.0 if all(abs(a - b) <= 1e-12 for a, b in zip(left, right)) else 0.0
    return sum(a * b for a, b in zip(left_centered, right_centered)) / denom


def _moving_average(values: Sequence[float], width: int = 5) -> list[float]:
    if len(values) < width:
        return list(values)
    return [sum(values[index : index + width]) / width for index in range(len(values) - width + 1)]


def _curve_metrics(reference: Sequence[float], candidate: Sequence[float]) -> dict[str, float]:
    mse = sum((a - b) ** 2 for a, b in zip(reference, candidate, strict=True)) / len(reference)
    rms_reference = math.sqrt(sum(value * value for value in reference) / len(reference))
    nrmse = math.sqrt(mse) / max(rms_reference, 1e-12)
    tail = min(10, len(reference))
    ref_tail = sum(reference[-tail:]) / tail
    cand_tail = sum(candidate[-tail:]) / tail
    symmetric_tail = 2.0 * abs(ref_tail - cand_tail) / max(abs(ref_tail) + abs(cand_tail), 1e-12)
    return {
        "normalized_rmse": nrmse,
        "moving_average_pearson": _pearson(_moving_average(reference), _moving_average(candidate)),
        "tail_symmetric_relative_difference": symmetric_tail,
    }


def _grid_snap_matches(exact: Mapping[str, Any], grid: Mapping[str, Any]) -> bool:
    low = float(grid["low"])
    high = float(grid["high"])
    continuous = float(exact["continuous_base"])
    position = (continuous - low) / (high - low) * 999
    index = min(999, max(0, int(math.floor(position + 0.5))))
    expected = float(torch.linspace(low, high, 1000, dtype=torch.float32)[index])
    return int(grid["grid_index"]) == index and float(grid["actual_base"]) == expected


def _summary_metric(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    left_coordinates = list(left.get("coordinates", ()))
    right_coordinates = list(right.get("coordinates", ()))
    coordinates_match = left_coordinates == right_coordinates
    left_values = [float(value) for value in left.get("values", ())]
    right_values = [float(value) for value in right.get("values", ())]
    if not coordinates_match or len(left_values) != len(right_values) or not left_values:
        return {
            "coordinates_match": coordinates_match,
            "sample_count": min(len(left_values), len(right_values)),
            "cosine": None,
            "normalized_l2": None,
            "norm_ratio": None,
            "max_abs": None,
        }
    left_norm = math.sqrt(sum(value * value for value in left_values))
    right_norm = math.sqrt(sum(value * value for value in right_values))
    diff_norm = math.sqrt(
        sum((a - b) ** 2 for a, b in zip(left_values, right_values, strict=True))
    )
    denominator = max(left_norm * right_norm, 1e-24)
    cosine = sum(a * b for a, b in zip(left_values, right_values, strict=True)) / denominator
    return {
        "coordinates_match": True,
        "sample_count": len(left_values),
        "cosine": cosine,
        "normalized_l2": diff_norm / max(left_norm, 1e-12),
        "norm_ratio": right_norm / max(left_norm, 1e-12),
        "max_abs": max(abs(a - b) for a, b in zip(left_values, right_values, strict=True)),
    }


def _walk_state_summaries(
    left: Any, right: Any, *, path: str = "state"
) -> list[tuple[str, Mapping[str, Any], Mapping[str, Any]]]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if "coordinates" in left and "values" in left and "coordinates" in right and "values" in right:
            return [(path, left, right)]
        result = []
        for key in sorted(set(left) & set(right)):
            result.extend(_walk_state_summaries(left[key], right[key], path=f"{path}.{key}"))
        return result
    if isinstance(left, list) and isinstance(right, list):
        result = []
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=False)):
            result.extend(
                _walk_state_summaries(left_item, right_item, path=f"{path}[{index}]")
            )
        return result
    return []


def _state_structure_matches(left: Any, right: Any) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _state_structure_matches(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _state_structure_matches(a, b) for a, b in zip(left, right, strict=True)
        )
    return type(left) is type(right)


def _identity_contract(
    reference_start: dict[int, dict[str, Any]], candidate_start: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    if not reference_start or not candidate_start:
        return {"available": False, "common_runtime_identity_match": None, "seed_match": None}
    common_keys = (
        "upstream_lightx2v",
        "model_config_sha256",
        "prompt_metadata_sha256",
        "attention_backend",
        "checkpoint_segment",
    )
    mismatches = []
    seed_match = True
    for rank in sorted(set(reference_start) & set(candidate_start)):
        left = reference_start[rank]
        right = candidate_start[rank]
        seed_match &= int(left["seed"]) == int(right["seed"])
        left_identity = left.get("identity", {})
        right_identity = right.get("identity", {})
        for key in common_keys:
            if left_identity.get(key) != right_identity.get(key):
                mismatches.append(
                    {
                        "rank": rank,
                        "key": key,
                        "reference": left_identity.get(key),
                        "candidate": right_identity.get(key),
                    }
                )
    reference_identity = reference_start[min(reference_start)].get("identity", {})
    candidate_identity = candidate_start[min(candidate_start)].get("identity", {})
    reference_consistent = all(
        row.get("identity", {}) == reference_identity for row in reference_start.values()
    )
    candidate_consistent = all(
        row.get("identity", {}) == candidate_identity for row in candidate_start.values()
    )
    return {
        "available": True,
        "common_runtime_identity_match": not mismatches,
        "seed_match": seed_match,
        "reference_rank_identity_consistent": reference_consistent,
        "candidate_rank_identity_consistent": candidate_consistent,
        "reference_identity": reference_identity,
        "candidate_identity": candidate_identity,
        "system_policy_note": (
            "activation/offload/kernel environment is recorded per arm but is not a cross-arm "
            "mathematical-identity gate"
        ),
        "mismatches": mismatches,
    }


def compare_trajectory_roots(
    reference_root: Path, candidate_root: Path, *, expected_cycles: int = 50
) -> dict[str, Any]:
    reference = _read_cycle_rows(Path(reference_root))
    candidate = _read_cycle_rows(Path(candidate_root))
    reference_start = _read_start_rows(Path(reference_root))
    candidate_start = _read_start_rows(Path(candidate_root))
    ranks_match = set(reference) == set(candidate)
    declared_world_size = len(reference)
    if reference_start:
        declared_world_size = int(next(iter(reference_start.values()))["world_size"])
    elif reference:
        declared_world_size = int(next(iter(reference.values()))[0].get("world_size", len(reference)))
    complete_rank_set = set(reference) == set(range(declared_world_size))
    cycle_counts_match = all(len(rows) == expected_cycles for rows in reference.values()) and all(
        len(rows) == expected_cycles for rows in candidate.values()
    )
    reference_variant = next(iter(reference.values()))[0]["variant"]
    candidate_variant = next(iter(candidate.values()))[0]["variant"]
    approximation = reference_variant == "exact" and candidate_variant == "grid1000"
    samples_match = True
    operation_keys_match = True
    sigma_inventory_match = True
    continuous_match = True
    actual_match = True
    grid_snap_match = True
    cycle_ordinals_match = True
    compute_census_match = True
    update_versions_match = True
    state_rows = []
    state_coordinates_match = True
    state_structure_match = True
    for rank in sorted(set(reference) & set(candidate)):
        for left, right in zip(reference[rank], candidate[rank]):
            cycle_ordinals_match &= int(left["cycle"]) == int(right["cycle"])
            left_samples = [item["identity"] for item in left["matched_compute"]["samples"]]
            right_samples = [item["identity"] for item in right["matched_compute"]["samples"]]
            samples_match &= left_samples == right_samples
            for key in (
                "fixed_end_step_idx",
                "forward_counts",
                "grad_forward_counts",
                "backward_counts",
            ):
                compute_census_match &= left["matched_compute"].get(key) == right[
                    "matched_compute"
                ].get(key)
            update_versions_match &= left.get("versions") == right.get("versions")
            sigma_inventory_match &= len(left["sigmas"]) == len(right["sigmas"]) == 6
            for left_sigma, right_sigma in zip(left["sigmas"], right["sigmas"]):
                operation_keys_match &= left_sigma["operation_key"] == right_sigma["operation_key"]
                continuous_match &= left_sigma["continuous_base"] == right_sigma["continuous_base"]
                actual_match &= left_sigma["actual_base"] == right_sigma["actual_base"]
                if approximation:
                    grid_snap_match &= _grid_snap_matches(left_sigma, right_sigma)
            left_anchor = left.get("state_anchor")
            right_anchor = right.get("state_anchor")
            state_structure_match &= (left_anchor is None) == (right_anchor is None)
            if left_anchor is not None and right_anchor is not None:
                state_structure_match &= _state_structure_matches(
                    left_anchor, right_anchor
                )
                for state_path, left_summary, right_summary in _walk_state_summaries(
                    left_anchor, right_anchor
                ):
                    metrics = _summary_metric(left_summary, right_summary)
                    state_coordinates_match &= bool(metrics["coordinates_match"])
                    state_rows.append(
                        {
                            "rank": rank,
                            "cycle": int(left["cycle"]),
                            "path": state_path,
                            **metrics,
                        }
                    )
    curves = {
        "student_dmd": (_curve(reference, "dmd_local"), _curve(candidate, "dmd_local")),
        **{
            f"fake_{index}": (
                _curve(reference, "fake_local", index),
                _curve(candidate, "fake_local", index),
            )
            for index in range(5)
        },
    }
    curve_metrics = {
        name: _curve_metrics(left, right) for name, (left, right) in curves.items()
    }
    soft_curve_pass = all(
        metrics["normalized_rmse"] <= 0.10
        and metrics["moving_average_pearson"] >= 0.95
        and metrics["tail_symmetric_relative_difference"] <= 0.10
        for metrics in curve_metrics.values()
    )
    comparable_state_rows = [row for row in state_rows if row["cosine"] is not None]
    state_summary = {
        "available": bool(comparable_state_rows),
        "comparison_count": len(state_rows),
        "coordinates_match": state_coordinates_match,
        "structure_match": state_structure_match,
        "minimum_cosine": min(
            (float(row["cosine"]) for row in comparable_state_rows), default=None
        ),
        "maximum_normalized_l2": max(
            (float(row["normalized_l2"]) for row in comparable_state_rows), default=None
        ),
        "rows": state_rows,
    }
    soft_state_pass = not comparable_state_rows or (
        state_coordinates_match
        and state_summary["minimum_cosine"] >= 0.999
        and state_summary["maximum_normalized_l2"] <= 0.05
    )
    identity = _identity_contract(reference_start, candidate_start)
    contract = {
        "ranks_match": ranks_match,
        "complete_rank_set": complete_rank_set,
        "cycle_counts_match": cycle_counts_match,
        "cycle_ordinals_match": cycle_ordinals_match,
        "sample_identity_match": samples_match,
        "compute_census_match": compute_census_match,
        "logical_update_versions_match": update_versions_match,
        "operation_keys_match": operation_keys_match,
        "sigma_inventory_match": sigma_inventory_match,
        "operation_keyed_continuous_sigma_match": continuous_match,
        "actual_sigma_match": actual_match if not approximation else None,
        "grid_snap_match": grid_snap_match if approximation else None,
        "state_coordinates_match": state_coordinates_match,
        "state_structure_match": state_structure_match,
        "common_runtime_identity_match": identity["common_runtime_identity_match"],
        "seed_match": identity["seed_match"],
        "reference_rank_identity_consistent": identity.get(
            "reference_rank_identity_consistent"
        ),
        "candidate_rank_identity_consistent": identity.get(
            "candidate_rank_identity_consistent"
        ),
    }
    contract_pass = all(value is not False for value in contract.values())
    if not contract_pass:
        status = "CONTRACT_MISMATCH_50C_COMPLETED"
    elif approximation:
        status = (
            "APPROXIMATION_CURVE_CLOSE_50C_SINGLE_SEED"
            if soft_curve_pass and soft_state_pass
            else "APPROXIMATION_TRAJECTORY_REVIEW_50C_COMPLETED"
        )
    else:
        status = (
            "TRAJECTORY_MATCHED_50C_SINGLE_SEED"
            if soft_curve_pass and soft_state_pass
            else "TRAJECTORY_REVIEW_50C_COMPLETED"
        )
    return {
        "schema": "h3-trajectory-comparison/v1",
        "status": status,
        "comparison_kind": (
            "exact_vs_grid1000_approximation" if approximation else "same_semantics"
        ),
        "reference_variant": reference_variant,
        "candidate_variant": candidate_variant,
        "expected_cycles": expected_cycles,
        "contract": contract,
        "identity": identity,
        "soft_thresholds": {
            "loss_normalized_rmse_max": 0.10,
            "loss_moving_average_pearson_min": 0.95,
            "loss_tail_symmetric_relative_difference_max": 0.10,
            "state_cosine_min": 0.999,
            "state_normalized_l2_max": 0.05,
            "note": "soft final classification only; never an online early-stop gate",
        },
        "loss_curves": {
            name: {"reference": left, "candidate": right, "metrics": curve_metrics[name]}
            for name, (left, right) in curves.items()
        },
        "state_sketch": state_summary,
    }
