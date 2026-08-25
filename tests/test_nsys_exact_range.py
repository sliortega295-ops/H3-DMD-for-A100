from __future__ import annotations

import os

from h3_a100.nsys_range import enabled, exact_cycle_range


def test_nsys_exact_range_is_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("H3_NSYS_EXACT_RANGE", raising=False)
    assert enabled() is False
    with exact_cycle_range(7):
        pass


def test_nsys_exact_range_switch_is_independent(monkeypatch) -> None:
    monkeypatch.setenv("H3_NSYS_EXACT_RANGE", "yes")
    assert enabled() is True
    monkeypatch.setenv("H3_NSYS_EXACT_RANGE", "off")
    assert enabled() is False
