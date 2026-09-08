"""Tests for gb32mn (Gibbons shelf)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.gb32mn import gibbons_runs_mean


def test_gb32mn_basic():
    assert gibbons_runs_mean(5, 5)["mean"] == pytest.approx(1 + 50 / 10)


def test_gb32mn_edge():
    with pytest.raises(ValueError):
        gibbons_runs_mean(0, 5)
