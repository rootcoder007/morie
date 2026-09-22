"""Tests for gb_cons.gibbons_consistency."""

from morie.fn import _array_core as np

from morie.fn.gb_cons import gibbons_consistency


def _expected_power(nvals, effect, alpha=0.05):
    from scipy import stats
    import math
    za = stats.norm.ppf(1.0 - alpha)
    return [1.0 - stats.norm.cdf(za - math.sqrt(v) * effect) for v in nvals]


def test_gb_cons_basic():
    """Test basic functionality with a documented sequence of sample sizes and a positive effect."""
    rng = np.random.default_rng(42)
    nvals = [5, 10, 20, 40, 80]
    effect = 0.3
    result = gibbons_consistency(nvals, effect)
    assert isinstance(result, dict)
    expected = _expected_power(nvals, effect)
    assert "power" in result
    assert len(result["power"]) == len(nvals)
    for got, exp in zip(result["power"], expected):
        assert abs(got - exp) < 1e-12
    assert result["consistent"] == 1
    assert result["monotone"] == 1
    assert "limit" in result
    assert abs(result["limit"] - expected[-1]) < 1e-12
    assert result["effect"] == effect
    assert "method" in result


def test_gb_cons_zero_effect():
    """With effect == 0 the test is not consistent."""
    nvals = [5, 10, 20, 40]
    effect = 0.0
    result = gibbons_consistency(nvals, effect)
    assert isinstance(result, dict)
    assert "power" in result
    expected = _expected_power(nvals, effect)
    for got, exp in zip(result["power"], expected):
        assert abs(got - exp) < 1e-12
    assert result["consistent"] == 0


def test_gb_cons_edge():
    """Test edge cases: single sample size and custom alpha."""
    nvals = [1]
    effect = 0.5
    result = gibbons_consistency(nvals, effect, alpha=0.01)
    assert isinstance(result, dict)
    assert "power" in result
    assert len(result["power"]) == 1
    expected = _expected_power(nvals, effect, alpha=0.01)
    assert abs(result["power"][0] - expected[0]) < 1e-12
