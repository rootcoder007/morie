"""Tests for fzt15.fauzi_thm1_5_consistency_mgkde."""

from morie.fn import _array_core as np

from morie.fn.fzt15 import fauzi_thm1_5_consistency_mgkde


def test_fzt15_basic():
    """Test basic functionality: interior variance with n provided."""
    rng = np.random.default_rng(42)
    varh = float(rng.normal(0.05, 0.005))
    var4h = float(rng.normal(0.02, 0.002))
    cov = float(rng.normal(0.01, 0.001))
    n = 500
    result = fauzi_thm1_5_consistency_mgkde(varh, var4h, cov, n=n, boundary=False)
    assert isinstance(result, dict)
    expected_variance = 4.0 * varh + var4h - 4.0 * cov
    assert abs(result["variance"] - expected_variance) < 1e-12
    expected_hopt = float(n) ** (-4.0 / 9.0)
    expected_mserate = float(n) ** (-8.0 / 9.0)
    assert abs(result["hopt"] - expected_hopt) < 1e-12
    assert abs(result["mserate"] - expected_mserate) < 1e-12
    assert result["region"] == "interior"
    assert "method" in result
    assert result["method"] == "modified gamma KDE variance (Theorem 1.5)"


def test_fzt15_boundary():
    """Test boundary rates."""
    rng = np.random.default_rng(7)
    varh = float(rng.normal(0.05, 0.005))
    var4h = float(rng.normal(0.02, 0.002))
    cov = float(rng.normal(0.01, 0.001))
    n = 500
    result = fauzi_thm1_5_consistency_mgkde(varh, var4h, cov, n=n, boundary=True)
    assert isinstance(result, dict)
    expected_variance = 4.0 * varh + var4h - 4.0 * cov
    assert abs(result["variance"] - expected_variance) < 1e-12
    expected_hopt = float(n) ** (-4.0 / 11.0)
    expected_mserate = float(n) ** (-8.0 / 11.0)
    assert abs(result["hopt"] - expected_hopt) < 1e-12
    assert abs(result["mserate"] - expected_mserate) < 1e-12
    assert result["region"] == "boundary"


def test_fzt15_no_n():
    """When n is omitted, hopt/mserate are NaN but variance still computed."""
    varh = 0.05
    var4h = 0.02
    cov = 0.01
    result = fauzi_thm1_5_consistency_mgkde(varh, var4h, cov, n=None, boundary=False)
    assert isinstance(result, dict)
    expected_variance = 4.0 * varh + var4h - 4.0 * cov
    assert abs(result["variance"] - expected_variance) < 1e-12
    assert result["hopt"] != result["hopt"]  # NaN
    assert result["mserate"] != result["mserate"]  # NaN


def test_fzt15_edge():
    """Edge case: n=1 must not raise and must produce the expected rate values."""
    varh = 0.05
    var4h = 0.02
    cov = 0.01
    result = fauzi_thm1_5_consistency_mgkde(varh, var4h, cov, n=1, boundary=False)
    assert isinstance(result, dict)
    expected_variance = 4.0 * varh + var4h - 4.0 * cov
    assert abs(result["variance"] - expected_variance) < 1e-12
    assert abs(result["hopt"] - 1.0) < 1e-12
    assert abs(result["mserate"] - 1.0) < 1e-12
