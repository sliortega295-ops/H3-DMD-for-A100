import json
from pathlib import Path

import pytest
import torch

from h3_a100.trajectory import (
    COARSE_SCHEMA,
    OperationKeyedSigmaSampler,
    TrajectoryRecorder,
    compare_trajectory_roots,
    reset_trajectory_rng,
    rng_state_receipt,
)


def test_operation_keyed_sigma_exact_and_grid_share_continuous_draw():
    exact = OperationKeyedSigmaSampler(seed=20260817, rank=7, variant="exact")
    grid = OperationKeyedSigmaSampler(seed=20260817, rank=7, variant="grid1000")
    torch.manual_seed(1234)
    before = torch.random.get_rng_state().clone()

    exact_sample = exact.sample(cycle=12, slot=3, low=0.02, high=0.98)
    grid_sample = grid.sample(cycle=12, slot=3, low=0.02, high=0.98)

    assert torch.equal(before, torch.random.get_rng_state())
    assert exact_sample.continuous_base == grid_sample.continuous_base
    assert exact_sample.actual_base == exact_sample.continuous_base
    assert grid_sample.grid_index is not None
    expected_grid = torch.linspace(0.02, 0.98, 1000, dtype=torch.float32)
    assert grid_sample.actual_base == float(expected_grid[grid_sample.grid_index])
    assert grid_sample.snap_abs_error == pytest.approx(
        abs(grid_sample.actual_base - grid_sample.continuous_base)
    )
    assert exact_sample.operation_key == "rank=7/cycle=12/fake_2/renoise_sigma"
    assert grid_sample.operation_key == exact_sample.operation_key


def test_operation_keyed_sigma_is_rank_and_operation_qualified():
    sampler = OperationKeyedSigmaSampler(seed=11, rank=0, variant="exact")
    values = {
        sampler.sample(cycle=cycle, slot=slot, low=0.02, high=0.98).continuous_base
        for cycle in range(2)
        for slot in range(6)
    }
    other_rank = OperationKeyedSigmaSampler(seed=11, rank=1, variant="exact")
    assert len(values) == 12
    assert (
        other_rank.sample(cycle=0, slot=0, low=0.02, high=0.98).continuous_base
        != sampler.sample(cycle=0, slot=0, low=0.02, high=0.98).continuous_base
    )


def _recorder(tmp_path: Path, *, variant: str = "exact") -> TrajectoryRecorder:
    return TrajectoryRecorder(
        output_root=tmp_path,
        rank=0,
        world_size=1,
        seed=9,
        variant=variant,
        expected_cycles=2,
        anchors=(1, 2),
        samples_per_tensor=3,
        identity={"source_head": "abc", "config_sha256": "def"},
    )


def test_recorder_writes_append_only_cycle_receipts_and_state_sketch(tmp_path):
    recorder = _recorder(tmp_path)
    student = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    fake = torch.nn.Parameter(torch.tensor([5.0, 6.0, 7.0, 8.0]))
    student_optimizer = torch.optim.AdamW([student], lr=0.1)
    fake_optimizer = torch.optim.AdamW([fake], lr=0.1)
    named = {"student.weight": student, "fake.weight": fake}
    recorder.start_run(
        current_iter=0,
        max_train_iters=2,
        named_parameters=named,
        student_parameters=[student],
        fake_parameters=[fake],
        rng_reset=reset_trajectory_rng(seed=9, rank=0),
    )

    for cycle in range(2):
        recorder.begin_cycle(cycle)
        for _ in range(6):
            recorder.sample_renoise_sigmas(
                low=0.02,
                high=0.98,
                video_shift=6.0,
                audio_shift=3.0,
                device=torch.device("cpu"),
            )
            pre = rng_state_receipt()
            recorder.note_stochastic_pair(
                kind="rollout_initial",
                tensors=(torch.randn(16), torch.randn(8)),
                pre_rng=pre,
            )
            pre = rng_state_receipt()
            recorder.note_stochastic_pair(
                kind="score_noise",
                tensors=(torch.randn(16), torch.randn(8)),
                pre_rng=pre,
            )
        student.grad = torch.ones_like(student)
        recorder.capture_gradient("student", [student])
        student_optimizer.step()
        for _ in range(5):
            fake.grad = torch.full_like(fake, 2.0)
            recorder.capture_gradient("fake", [fake])
            fake_optimizer.step()
        snapshot = {
            "fixed_end_step_idx": 3,
            "forward_counts": {"student": 24, "fake": 6, "teacher": 1},
            "grad_forward_counts": {"student": 1, "fake": 5, "teacher": 0},
            "backward_counts": {"student": 1, "fake": 5},
            "samples": [{"stage": f"s{i}", "identity": f"r0-c{cycle}-s{i}"} for i in range(6)],
        }
        recorder.finish_cycle(
            cycle=cycle,
            local_dmd=[torch.tensor(1.0 + cycle)],
            local_fake=[torch.tensor(float(i + cycle)) for i in range(5)],
            world_dmd=1.0 + cycle,
            world_fake=2.0 + cycle,
            matched_snapshot=snapshot,
            student_parameters=[student],
            fake_parameters=[fake],
            student_optimizer=student_optimizer,
            fake_optimizer=fake_optimizer,
            student_scheduler_steps=cycle + 1,
            fake_scheduler_steps=(cycle + 1) * 5,
            student_optimizer_updates=cycle + 1,
            fake_optimizer_updates=(cycle + 1) * 5,
        )
    recorder.finish_run()

    rows = [json.loads(line) for line in (tmp_path / "rank_000.trajectory.jsonl").read_text().splitlines()]
    assert [row["record_type"] for row in rows] == ["run_start", "cycle", "cycle", "run_end"]
    assert rows[1]["schema"] == COARSE_SCHEMA
    assert rows[1]["losses"]["fake_local"] == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert len(rows[1]["sigmas"]) == 6
    assert rows[1]["state_anchor"]["student"]["gradient"]["sample_count"] == 3
    assert rows[2]["versions"] == {
        "student_optimizer_updates": 2,
        "fake_optimizer_updates": 10,
        "student_scheduler_steps": 2,
        "fake_scheduler_steps": 10,
        "ema_updates": 0,
    }


def test_recorder_refuses_to_overwrite_existing_rank_receipt(tmp_path):
    path = tmp_path / "rank_000.trajectory.jsonl"
    path.write_text("old evidence\n")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        _recorder(tmp_path)


def _summary(value: float = 1.0):
    return {
        "sample_count": 1,
        "l2": abs(value),
        "mean": value,
        "max_abs": abs(value),
        "sha256": f"sample-{value}",
        "coordinates": ["p[0]"],
        "values": [value],
    }


def _arm(variant: str):
    if variant == "exact":
        return {
            "arm_id": "h3-exact-continuous-boundary-cpu/v1",
            "attention_backend": "_flash_3_hub",
            "checkpoint_segment": 1,
            "activation_policy": "checkpoint_boundary_cpu",
            "fa3_replay_cache": "disabled",
        }
    return {
        "arm_id": "h3-grid1000-iteration418/v1",
        "attention_backend": "_flash_3_hub",
        "checkpoint_segment": 1,
        "activation_policy": "none",
        "fa3_replay_cache_blocks": "0-49",
        "fa3_replay_cache_storage": "cpu_staged",
        "fa3_replay_cache_max_d2h_inflight": 2,
        "fa3_replay_cache_trim_before_backward": True,
    }


def _write_root(root: Path, *, variant: str, offset: float = 0.0):
    root.mkdir()
    for rank in range(16):
        sampler = OperationKeyedSigmaSampler(seed=9, rank=rank, variant=variant)
        identity = {
            "source_head": f"head-{variant}",
            "upstream_lightx2v": "upstream",
            "python": "3.10",
            "torch": "2.10",
            "torch_cuda": "12.8",
            "diffusers": "x",
            "peft": "y",
            "model_config_sha256": "model",
            "prompt_metadata_sha256": "prompts",
            "attention_backend": "_flash_3_hub",
            "checkpoint_segment": 1,
            "activation_policy": _arm(variant)["activation_policy"],
            "grid_manifest_sha256": "grid-hash" if variant == "grid1000" else None,
            "arm_contract": _arm(variant),
        }
        start = {
            "schema": COARSE_SCHEMA,
            "record_type": "run_start",
            "rank": rank,
            "world_size": 16,
            "variant": variant,
            "seed": 9,
            "expected_cycles": 2,
            "anchors": [1, 2],
            "post_setup_rng_reset": {"rank_seed": 9 + rank, "state": {"sha": f"r{rank}"}},
            "identity": identity,
            "initial_state": {"student": _summary(), "fake": _summary()},
        }
        rows = [start]
        for cycle in range(2):
            sigmas = [
                sampler.sample(cycle=cycle, slot=slot, low=0.02, high=0.98).as_dict()
                for slot in range(6)
            ]
            stochastic = {
                kind: [
                    {
                        "operation": operation,
                        "pre_rng": {"sha": f"pre-{cycle}-{slot}"},
                        "post_rng": {"sha": f"post-{cycle}-{slot}"},
                        "tensor": _summary(float(slot + 1)),
                    }
                    for slot, operation in enumerate(
                        ("student", "fake_0", "fake_1", "fake_2", "fake_3", "fake_4")
                    )
                ]
                for kind in ("rollout_initial", "score_noise")
            }
            role_state = {
                "student": {
                    "gradient": _summary(),
                    "parameter_delta": _summary(),
                    "adam_exp_avg": _summary(),
                    "adam_exp_avg_sq": _summary(),
                },
                "fake": {
                    "gradient": [_summary() for _ in range(5)],
                    "parameter_delta": _summary(),
                    "adam_exp_avg": _summary(),
                    "adam_exp_avg_sq": _summary(),
                },
            }
            rows.append(
                {
                    "schema": COARSE_SCHEMA,
                    "record_type": "cycle",
                    "cycle": cycle + 1,
                    "rank": rank,
                    "world_size": 16,
                    "variant": variant,
                    "losses": {
                        "dmd_local": [1.0 + cycle + offset],
                        "fake_local": [2.0 + cycle + offset] * 5,
                    },
                    "sigmas": sigmas,
                    "rng": {
                        "cycle_start": {"sha": f"start-{cycle}"},
                        "cycle_end": {"sha": f"end-{cycle}"},
                        "stochastic": stochastic,
                    },
                    "matched_compute": {
                        "fixed_end_step_idx": 3,
                        "forward_counts": {"student": 24, "fake": 6, "teacher": 1},
                        "grad_forward_counts": {"student": 1, "fake": 5, "teacher": 0},
                        "backward_counts": {"student": 1, "fake": 5},
                        "samples": [
                            {
                                "stage": stage,
                                "identity": f"rank-{rank}-cycle-{cycle}-{stage}",
                            }
                            for stage in ("student", "fake_0", "fake_1", "fake_2", "fake_3", "fake_4")
                        ],
                    },
                    "versions": {
                        "student_optimizer_updates": cycle + 1,
                        "fake_optimizer_updates": (cycle + 1) * 5,
                        "student_scheduler_steps": cycle + 1,
                        "fake_scheduler_steps": (cycle + 1) * 5,
                        "ema_updates": 0,
                    },
                    "state_anchor": role_state,
                }
            )
        rows.append(
            {
                "schema": COARSE_SCHEMA,
                "record_type": "run_end",
                "rank": rank,
                "world_size": 16,
                "variant": variant,
                "completed_cycles": 2,
                "status": "COMPLETE",
            }
        )
        (root / f"rank_{rank:03d}.trajectory.jsonl").write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
        )
        if rank == 0:
            (root / "trajectory_manifest.json").write_text(
                json.dumps(start, indent=2, sort_keys=True) + "\n"
            )


def test_exact_vs_grid_comparison_validates_continuous_to_snap_alignment(tmp_path):
    exact = tmp_path / "exact"
    grid = tmp_path / "grid"
    _write_root(exact, variant="exact")
    _write_root(grid, variant="grid1000", offset=0.001)
    report = compare_trajectory_roots(exact, grid, expected_cycles=2)
    assert report["comparison_kind"] == "exact_vs_grid1000_approximation"
    assert report["contract"]["operation_keyed_continuous_sigma_match"] is True
    assert report["contract"]["grid_snap_match"] is True
    assert report["contract"]["post_setup_rng_and_stochastic_receipts_match"] is True
    assert report["contract"]["initial_trainable_state_match"] is True
    assert report["status"].startswith("APPROXIMATION_")


def test_comparison_fails_closed_on_missing_rank_or_end(tmp_path):
    exact = tmp_path / "exact"
    grid = tmp_path / "grid"
    _write_root(exact, variant="exact")
    _write_root(grid, variant="grid1000")
    (grid / "rank_015.trajectory.jsonl").unlink()
    with pytest.raises(RuntimeError, match="world is incomplete"):
        compare_trajectory_roots(exact, grid, expected_cycles=2)


def test_comparison_fails_closed_on_missing_end_and_manifest(tmp_path):
    exact = tmp_path / "exact"
    grid = tmp_path / "grid"
    _write_root(exact, variant="exact")
    _write_root(grid, variant="grid1000")
    path = grid / "rank_007.trajectory.jsonl"
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(RuntimeError, match="records="):
        compare_trajectory_roots(exact, grid, expected_cycles=2)
    path.write_text("\n".join(lines) + "\n")
    (grid / "trajectory_manifest.json").unlink()
    with pytest.raises(RuntimeError, match="manifest is missing"):
        compare_trajectory_roots(exact, grid, expected_cycles=2)


def test_comparison_rejects_empty_adam_and_synthetic_update_counter(tmp_path):
    exact = tmp_path / "exact"
    grid = tmp_path / "grid"
    _write_root(exact, variant="exact")
    _write_root(grid, variant="grid1000")
    path = grid / "rank_003.trajectory.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1]["state_anchor"]["student"]["adam_exp_avg"] = _summary()
    rows[1]["state_anchor"]["student"]["adam_exp_avg"]["sample_count"] = 0
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(RuntimeError, match="Adam|adam"):
        compare_trajectory_roots(exact, grid, expected_cycles=2)
    rows[1]["state_anchor"]["student"]["adam_exp_avg"] = _summary()
    rows[2]["versions"]["fake_optimizer_updates"] = 9
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(RuntimeError, match="update counters"):
        compare_trajectory_roots(exact, grid, expected_cycles=2)


def test_post_setup_rng_reset_erases_setup_consumption():
    torch.manual_seed(1)
    _ = torch.randn(17)
    first = reset_trajectory_rng(seed=123, rank=7)
    first_draw = torch.randn(8)
    torch.manual_seed(999)
    _ = torch.randn(31)
    second = reset_trajectory_rng(seed=123, rank=7)
    second_draw = torch.randn(8)
    assert first == second
    assert torch.equal(first_draw, second_draw)


def test_trajectory_launcher_defaults_to_50_and_allows_bounded_canaries():
    source = (Path(__file__).parents[1] / "scripts" / "launch_trajectory_50c.sh").read_text()
    assert "H3_TRAJECTORY_CYCLES=${H3_TRAJECTORY_CYCLES:-50}" in source
    assert 'export H3_MAX_ITERS="${H3_TRAJECTORY_CYCLES}"' in source
    assert 'export H3_TRAJECTORY_EXPECTED_CYCLES="${H3_TRAJECTORY_CYCLES}"' in source
    assert "1) DEFAULT_TRAJECTORY_ANCHORS=1" in source
    assert "5) DEFAULT_TRAJECTORY_ANCHORS=1,5" in source
    assert "50) DEFAULT_TRAJECTORY_ANCHORS=1,10,25,50" in source
    assert "h3-exact-continuous-boundary-cpu/v1" in source
    assert "h3-grid1000-iteration418/v1" in source
    assert "H3_ACTIVATION_POLICY=checkpoint_boundary_cpu" in source
    assert "H3_ACTIVATION_POLICY=none" in source
    assert "H3_FA3_REPLAY_CACHE_BLOCKS=0-49" in source
    assert "H3_FA3_REPLAY_CACHE_STORAGE=cpu_staged" in source
    assert "H3_FA3_REPLAY_CACHE_MAX_D2H_INFLIGHT=2" in source
    assert "early_stop" not in source.lower()
    assert "cosine" not in source.lower()
