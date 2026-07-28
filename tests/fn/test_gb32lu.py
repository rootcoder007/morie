"""Tests for gb32lu (Gibbons shelf)."""

import numpy as np
import pytest

from morie.fn.gb32lu import gibbons_runs_up_down_recur


def test_gb32lu_basic():
    out = gibbons_runs_up_down_recur(n=6)
    assert out["mean"] == pytest.approx((2 * 6 - 1) / 3, abs=1e-12)


def test_gb32lu_edge():
    with pytest.raises(ValueError):
        gibbons_runs_up_down_recur(x=[1.0, 1.0, 2.0])  # ties
