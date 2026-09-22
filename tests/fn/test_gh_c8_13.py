"""Tests for gh_c8_13.ghosal_misspec_crt."""

from morie.fn import _array_core as np

from morie.fn.gh_c8_13 import ghosal_misspec_crt


def test_gh_c8_13_basic():
    """Test basic functionality."""
    p0 = (0.6, 0.3, 0.1)
    n = 2000
    seed = 42
    result = ghosal_misspec_crt(p0=p0, n=n, seed=seed)
    assert "estimate" in result
    estimate = np.asarray(result["estimate"], dtype=float)
    assert np.all(np.isfinite(estimate))
    # The KL projection t* equals p0[2] (the third component of p0)
    assert "kl_projection_t" in result
    t_star = np.asarray(result["kl_projection_t"], dtype=float)
    assert t_star == p0[2]
    # The posterior contraction estimate should be close to t_star
    assert abs(float(estimate) - p0[2]) < 0.05
    # Computed from formula: error_to_projection = |best_t - t_star|
    assert "error_to_projection" in result
    expected_error = abs(float(estimate) - p0[2])
    assert abs(float(result["error_to_projection"]) - expected_error) < 1e-9


def test_gh_c8_13_edge():
    """Test edge cases with deterministic seed and minimal n."""
    p0 = (0.5, 0.5, 0.0)
    result = ghosal_misspec_crt(p0=p0, n=1, seed=0)
    assert "estimate" in result
    assert "kl_projection_t" in result
    # With p0[2] = 0, t_star = 0, so the estimate should be small
    assert result["kl_projection_t"] == 0.0
    estimate = float(np.asarray(result["estimate"], dtype=float))
    assert np.isfinite(estimate)
