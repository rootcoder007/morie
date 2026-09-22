"""Tests for bsmed.bootstrap_mediation_ci."""

from morie.fn import _array_core as np

from morie.fn.bsmed import bootstrap_mediation_ci


def test_bsmed_basic():
    """Test basic functionality."""
    rng_x = np.random.default_rng(42)
    X = rng_x.normal(0, 1, 100)
    rng_m = np.random.default_rng(43)
    M = rng_m.normal(0, 1, 100)
    rng_y = np.random.default_rng(44)
    Y = rng_y.normal(0, 1, 100)
    result = bootstrap_mediation_ci(X, M, Y)
    assert isinstance(result, dict)
    # Documented return keys per the function's docstring
    assert "estimate" in result
    assert "boot_estimate" in result
    assert "se" in result
    assert "ci_lower" in result
    assert "ci_upper" in result
    assert "a" in result
    assert "b" in result
    assert "c_prime" in result
    assert "B" in result
    assert "n" in result
    assert "conf_level" in result
    # Sample-size bookkeeping
    assert int(result["n"]) == 100
    assert int(result["B"]) == 1000
    # conf_level is 1 - alpha with default alpha=0.05
    assert abs(result["conf_level"] - 0.95) < 1e-12
    # ci_lower <= estimate <= ci_upper (point estimate lies inside percentile CI)
    assert result["ci_lower"] <= result["estimate"] <= result["ci_upper"]
    # a*b must equal the reported point estimate
    assert abs(result["estimate"] - (result["a"] * result["b"])) < 1e-12


def test_bsmed_edge():
    """Test edge cases: equal-length 1-D inputs and key presence."""
    rng = np.random.default_rng(7)
    X = rng.normal(0, 1, 50)
    M = rng.normal(0, 1, 50)
    Y = rng.normal(0, 1, 50)
    result = bootstrap_mediation_ci(X, M, Y)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "boot_estimate" in result
    assert "se" in result
    assert "ci_lower" in result
    assert "ci_upper" in result
    assert result["ci_lower"] <= result["ci_upper"]
