"""Tests for gh_c14_9.ghosal_py_process."""

from morie.fn import _array_core as np

from morie.fn.gh_c14_9 import ghosal_py_process


def test_gh_c14_9_basic():
    """Test basic functionality: shape, key presence, finiteness of returned estimate."""
    n_terms = 400
    d = 0.3
    theta = 1.0
    seed = 42
    result = ghosal_py_process(n_terms=n_terms, d=d, theta=theta, seed=seed)

    # Documented return keys
    assert "estimate" in result
    assert "total_mass" in result
    assert "top_weights" in result
    assert "method" in result

    # The estimate is the first stick-breaking weight W[0].
    est = np.asarray(result["estimate"], dtype=float)
    assert np.all(np.isfinite(est))
    assert float(est) > 0.0
    assert float(est) < 1.0

    # top_weights should contain the first 10 weights, all finite and in [0, 1]
    top = np.asarray(result["top_weights"], dtype=float)
    assert len(top) == 10
    assert np.all(np.isfinite(top))
    assert np.all(top >= 0.0)
    assert np.all(top <= 1.0)

    # Independent computation of W[0] via the documented stick-breaking formula
    # W[0] = V[0]; V[0] ~ Beta(1 - d, theta + 1*d)
    # The expected estimate equals the 1 - V[1] term only in expectation for k=0.
    # We only check structural properties (range, finiteness) here.


def test_gh_c14_9_edge():
    """Edge case: minimum n_terms=1 should still produce valid outputs."""
    result = ghosal_py_process(n_terms=1, d=0.3, theta=1.0, seed=7)

    assert "estimate" in result
    assert "total_mass" in result
    assert "top_weights" in result

    est = float(np.asarray(result["estimate"], dtype=float))
    total = float(np.asarray(result["total_mass"], dtype=float))
    top = np.asarray(result["top_weights"], dtype=float)

    assert 0.0 < est < 1.0
    # For n_terms=1, total_mass equals the single weight.
    assert abs(total - est) < 1e-12
    assert len(top) == 1
    assert abs(float(top[0]) - est) < 1e-12
