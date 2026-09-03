import json
from pathlib import Path

import pytest
import torch

from h3_a100.trajectory import (
    COARSE_SCHEMA,
    OperationKeyedSigmaSampler,
    TrajectoryRecorder,
    compare_trajectory_roots,
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
        student.grad = torch.ones_like(student)
        fake.grad = torch.full_like(fake, 2.0)
        recorder.capture_gradient("student", [student])
        recorder.capture_gradient("fake", [fake])
        student_optimizer.step()
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


def _write_root(root: Path, *, variant: str, offset: float = 0.0):
    root.mkdir()
    sampler = OperationKeyedSigmaSampler(seed=9, rank=0, variant=variant)
    rows = []
    for cycle in range(2):
        sigmas = [sampler.sample(cycle=cycle, slot=slot, low=0.02, high=0.98).as_dict() for slot in range(6)]
        rows.append(
            {
                "schema": COARSE_SCHEMA,
                "record_type": "cycle",
                "cycle": cycle + 1,
                "rank": 0,
                "variant": variant,
                "losses": {
                    "dmd_local": [1.0 + cycle + offset],
                    "fake_local": [2.0 + cycle + offset] * 5,
                },
                "sigmas": sigmas,
                "matched_compute": {"samples": [{"identity": f"sample-{cycle}-{i}"} for i in range(6)]},
                "state_anchor": {},
            }
        )
    (root / "rank_000.trajectory.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
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
    assert report["status"].startswith("APPROXIMATION_")


def test_trajectory_launcher_defaults_to_50_and_allows_bounded_canaries():
    source = (Path(__file__).parents[1] / "scripts" / "launch_trajectory_50c.sh").read_text()
    assert "H3_TRAJECTORY_CYCLES=${H3_TRAJECTORY_CYCLES:-50}" in source
    assert 'export H3_MAX_ITERS="${H3_TRAJECTORY_CYCLES}"' in source
    assert 'export H3_TRAJECTORY_EXPECTED_CYCLES="${H3_TRAJECTORY_CYCLES}"' in source
    assert "1) DEFAULT_TRAJECTORY_ANCHORS=1" in source
    assert "5) DEFAULT_TRAJECTORY_ANCHORS=1,5" in source
    assert "50) DEFAULT_TRAJECTORY_ANCHORS=1,10,25,50" in source
    assert "early_stop" not in source.lower()
    assert "cosine" not in source.lower()
