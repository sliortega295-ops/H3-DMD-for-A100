from __future__ import annotations

from h3_a100.matched_contract import (
    EXPECTED_BACKWARD_COUNTS,
    EXPECTED_FORWARD_COUNTS,
    EXPECTED_GRAD_FORWARD_COUNTS,
    EXPECTED_SAMPLE_STAGES,
    MatchedCycleCensus,
    validate_global_snapshots,
)


def _sample(index: int):
    return {"meta": {"lmdb_path": "/cache/h3.lmdb", "row_index": index}}


def _complete(rank: int = 0):
    census = MatchedCycleCensus(expected_world_size=16)
    for _ in range(24):
        census.note_forward("student", grad_enabled=False)
    census.grad_forward_counts["student"] = 1
    for index in range(6):
        census.note_forward("fake", grad_enabled=index != 0)
    census.note_forward("teacher", grad_enabled=False)
    census.note_backward("student")
    for _ in range(5):
        census.note_backward("fake")
    for stage_index, stage in enumerate(EXPECTED_SAMPLE_STAGES):
        census.note_sample(stage, _sample(rank * 100 + stage_index))
    return census


def test_exact_application_inventory_passes():
    census = _complete()
    assert census.forward_counts == EXPECTED_FORWARD_COUNTS
    assert census.grad_forward_counts == EXPECTED_GRAD_FORWARD_COUNTS
    assert census.backward_counts == EXPECTED_BACKWARD_COUNTS
    assert census.validate_local() == []


def test_missing_student_forward_fails():
    census = _complete()
    census.forward_counts["student"] -= 1
    errors = census.validate_local()
    assert any("forward_counts" in error for error in errors)


def test_global_contract_requires_96_unique_samples():
    snapshots = [_complete(rank).snapshot() for rank in range(16)]
    assert validate_global_snapshots(
        snapshots,
        expected_world_size=16,
        require_unique_samples=True,
    ) == []

    snapshots[15]["samples"][5]["identity"] = snapshots[0]["samples"][0]["identity"]
    errors = validate_global_snapshots(
        snapshots,
        expected_world_size=16,
        require_unique_samples=True,
    )
    assert any("global sample reuse" in error for error in errors)
