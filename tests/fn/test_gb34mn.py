"""Tests for gb34mn (Gibbons shelf)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.gb34mn import gibbons_runs_ud_mean


def test_gb34mn_basic():
    out = gibbons_runs_ud_mean(10)
    assert out["mean"] == pytest.approx(19 / 3)
    assert out["var"] == pytest.approx(131 / 90)


def test_gb34mn_edge():
    with pytest.raises(ValueError):
        gibbons_runs_ud_mean(2)
