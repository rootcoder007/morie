"""Tests for grucl.gru_cell."""

import numpy as np

from morie.fn.grucl import gru_cell


def test_grucl_basic():
    """A GRU cell returns a hidden-state vector, not a scalar mean."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = gru_cell(x)
    assert "estimate" in result
    h = np.asarray(result["estimate"], dtype=float)
    # The gate nonlinearities are bounded, so every unit sits in (-1, 1)
    # and none may be NaN/inf.
    assert h.size >= 1
    assert np.all(np.isfinite(h))
    assert np.all(np.abs(h) <= 1.0)


def test_grucl_edge():
    """A length-1 input still produces a finite, bounded state."""
    result = gru_cell(np.array([42.0]))
    h = np.asarray(result["estimate"], dtype=float)
    assert np.all(np.isfinite(h))
    assert np.all(np.abs(h) <= 1.0)
