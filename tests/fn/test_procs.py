"""Tests for procs."""

import numpy as np
import pytest

from morie.fn.procs import procrustes_rotation


def test_procs_basic():
    rng = np.random.default_rng(4)
    A = rng.normal(size=(8, 2))
    th = 0.5
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    out = procrustes_rotation(A, A @ R.T)
    assert out["residual"] == pytest.approx(0.0, abs=1e-10)
    assert out["rotated"] == pytest.approx(A, abs=1e-10)


def test_procs_edge():
    A = np.zeros((5, 2))
    with pytest.raises(ValueError):
        procrustes_rotation(A, np.zeros((4, 2)))  # shape mismatch
