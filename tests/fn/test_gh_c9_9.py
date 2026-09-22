"""Tests for gh_c9_9.ghosal_wn_conj_crt."""

from morie.fn import _array_core as np

from morie.fn.gh_c9_9 import ghosal_wn_conj_crt


def test_gh_c9_9_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    lam = np.array([0.5, 1.0, 2.0, 1.5, 0.25])
    n = 10.0
    result = ghosal_wn_conj_crt(x, n, lam)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))
    # Posterior mean per coordinate: n*X_i / (n + 1/lambda_i)
    expected_mean = np.array(
        [n * xi / (n + 1.0 / li) for xi, li in zip(x.tolist(), lam.tolist())]
    )
    # Posterior variance per coordinate: 1 / (n + 1/lambda_i)
    expected_var = np.array(
        [1.0 / (n + 1.0 / li) for li in lam.tolist()]
    )
    got_mean = np.asarray(result["posterior_mean"], dtype=float)
    got_var = np.asarray(result["posterior_var"], dtype=float)
    assert np.allclose(got_mean, expected_mean)
    assert np.allclose(got_var, expected_var)
    # The scalar estimate corresponds to the first coordinate.
    assert np.isclose(float(result["estimate"]), expected_mean[0])
    assert result["method"].startswith("white-noise conjugate posterior")


def test_gh_c9_9_edge():
    """Test edge cases."""
    x = np.array([42.0])
    lam = np.array([4.0])
    n = 1.0
    result = ghosal_wn_conj_crt(x, n, lam)
    # With lambda = 4 and n = 1: posterior mean = 1*42 / (1 + 1/4) = 42 / 1.25
    expected_mean = 42.0 / (1.0 + 1.0 / 4.0)
    # Posterior variance = 1 / (1 + 1/4) = 0.8
    expected_var = 1.0 / (1.0 + 1.0 / 4.0)
    assert np.isclose(float(result["estimate"]), expected_mean)
    assert np.isclose(
        float(np.asarray(result["posterior_mean"])[0]), expected_mean
    )
    assert np.isclose(
        float(np.asarray(result["posterior_var"])[0]), expected_var
    )
