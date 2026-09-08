"""Tests for gb322 (Gibbons shelf)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.gb322 import gibbons_total_runs_dist


def test_gb322_basic():
    total = sum(gibbons_total_runs_dist(r, 4, 5)["pmf"] for r in range(2, 10))
    assert total == pytest.approx(1.0, abs=1e-12)


def test_gb322_edge():
    with pytest.raises(ValueError):
        gibbons_total_runs_dist(1, 4, 5)
