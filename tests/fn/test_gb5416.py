"""Tests for gb5416.gibbons_sign_sampsize."""

from morie.fn.gb5416 import gibbons_sign_sampsize


def test_gb5416_basic():
    """Test basic functionality against the literature formula."""
    theta = 0.2
    alpha = 0.05
    beta = 0.10
    result = gibbons_sign_sampsize(theta, alpha=alpha, beta=beta)
    assert isinstance(result, dict)
    # Documented return keys.
    for key in ("n", "n_raw", "root_n", "z_alpha", "z_beta", "theta", "method"):
        assert key in result
    # theta is echoed back.
    assert result["theta"] == theta
    # Sample size rounded up to next integer, with the worked example giving N = 20.
    assert result["n"] == 20
    assert isinstance(result["n"], int)
    # root_n is the positive square root of n_raw.
    assert result["root_n"] == (result["n_raw"] ** 0.5)


def test_gb5416_edge():
    """Test edge cases."""
    theta = 0.2
    alpha = 0.05
    beta = 0.10
    result = gibbons_sign_sampsize(theta, alpha=alpha, beta=beta)
    assert isinstance(result, dict)
    assert result["n"] >= 1
