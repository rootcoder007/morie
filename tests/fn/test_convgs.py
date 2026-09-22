"""Tests for convgs.convergent_validity."""

from morie.fn import _array_core as np

from morie.fn.convgs import convergent_validity


def test_convgs_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    loadings = rng.uniform(0.5, 0.95, 100)
    # residual variance = 1 - lambda^2 for standardised indicators
    residuals = [1.0 - lam * lam for lam in loadings]
    result = convergent_validity(loadings, residuals)
    assert isinstance(result, dict)
    # the function's documented return keys
    assert "estimate" in result
    assert "ave" in result
    assert "cr" in result
    assert "adequate" in result
    assert "n_items" in result
    assert result["n_items"] == len(loadings)

    # independent recomputation of AVE from the documented formula
    sl2 = sum(lam * lam for lam in loadings)
    sth = sum(residuals)
    expected_ave = sl2 / (sl2 + sth)
    sl = sum(loadings)
    expected_cr = (sl * sl) / (sl * sl + sth)
    assert result["ave"] == expected_ave
    assert result["estimate"] == expected_ave
    assert result["cr"] == expected_cr
    assert result["adequate"] == (1 if (expected_ave >= 0.5 and expected_cr >= 0.7) else 0)


def test_convgs_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    loadings = rng.uniform(0.5, 0.95, 100)
    # default residuals derived as 1 - lambda^2 (standardised indicators)
    result = convergent_validity(loadings)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "ave" in result
    assert "cr" in result
    assert "n_items" in result
    assert result["n_items"] == len(loadings)

    # when residuals are None, theta_i = 1 - lambda_i^2, so
    # AVE = sum lambda^2 / (sum lambda^2 + sum (1 - lambda^2)) = sum lambda^2 / p
    expected_ave_no_res = sum(lam * lam for lam in loadings) / len(loadings)
    assert result["ave"] == expected_ave_no_res
