"""Tests for gb32vr (Gibbons shelf)."""

import numpy as np
import pytest

from morie.fn.gb32vr import gibbons_runs_var


def test_gb32vr_basic():
    out = gibbons_runs_var(5, 5)
    assert out["var"] == pytest.approx(2 * 25 * 40 / (100 * 9))


def test_gb32vr_edge():
    with pytest.raises(ValueError):
        gibbons_runs_var(1, 1)
