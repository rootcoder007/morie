"""Tests for gh_c14_12.ghosal_pk_process."""

from morie.fn import _array_core as np

from morie.fn.gh_c14_12 import ghosal_pk_process


def test_gh_c14_12_basic():
    """Test basic functionality."""
    n_jumps = 500
    seed = 42
    result = ghosal_pk_process(n_jumps=n_jumps, seed=seed)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))
    # Independent re-computation of the estimate using the documented formula:
    # J_k ~ Gamma(shape=(3/n_jumps), scale=1), W_k = J_k / sum(J_k),
    # estimate = max(W_k).
    rng = np.random.default_rng(seed)
    J = [float(rng.gamma(1.0 / n_jumps * 3.0, 1.0))
         for _ in range(n_jumps)]
    T = sum(J)
    W = [j / T for j in J]
    expected_estimate = max(W)
    assert abs(float(result["estimate"]) - expected_estimate) < 1e-12


def test_gh_c14_12_edge():
    """Test edge cases."""
    n_jumps = 1
    seed = 42
    result = ghosal_pk_process(n_jumps=n_jumps, seed=seed)
    # With a single jump, the normalised mass is trivially 1 and equals the max.
    assert abs(float(result["estimate"]) - 1.0) < 1e-12
    assert abs(float(result["total_mass"]) - 1.0) < 1e-12
    assert result["method"].startswith("Poisson-Kingman")
