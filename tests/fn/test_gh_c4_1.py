"""Tests for gh_c4_1.ghosal_dp_def."""

from morie.fn import _array_core as np

from morie.fn.gh_c4_1 import ghosal_dp_def


def test_gh_c4_1_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_dp_def(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))
    # Result must also carry the full probability vector, the dirichlet
    # parameters used, and a method label, per the documented behaviour.
    assert "P" in result
    assert "dir_params" in result
    assert "method" in result
    p = np.asarray(result["P"], dtype=float)
    assert p.shape == (5,)
    assert np.all(p >= 0.0)
    assert abs(float(np.sum(p)) - 1.0) < 1e-12
    # Independently recompute expected estimate as the first component
    # of the normalized probability vector.
    expected_estimate = float(p[0])
    assert abs(float(result["estimate"]) - expected_estimate) < 1e-12


def test_gh_c4_1_edge():
    """Test edge case: a single-element partition base mass vector."""
    result = ghosal_dp_def(np.array([42.0]))
    # A single gamma draw normalized by itself must yield probability 1.0.
    assert abs(float(result["estimate"]) - 1.0) < 1e-12
    assert abs(float(np.sum(np.asarray(result["P"], dtype=float))) - 1.0) < 1e-12
