"""Tests for fzt53.fauzi_thm5_3_bdfree_normality."""

from morie.fn import _array_core as np

from morie.fn.fzt53 import fauzi_thm5_3_bdfree_normality


def test_fzt53_basic():
    """Test basic functionality with documented signature."""
    estimate = 0.4
    variance = 0.04
    bias = 0.0
    level = 0.95
    result = fauzi_thm5_3_bdfree_normality(
        estimate, variance, null=0.5, bias=bias, level=level
    )
    assert isinstance(result, dict)
    for key in ("statistic", "p_value", "lower", "upper", "se", "level", "method"):
        assert key in result

    se = np.sqrt(variance)
    centre = estimate - bias
    expected_stat = (centre - 0.5) / se
    assert np.isclose(result["statistic"], expected_stat)
    assert np.isclose(result["se"], se)
    assert np.isclose(result["level"], level)

    from scipy.stats import norm as _norm

    expected_p = 2.0 * (1.0 - _norm.cdf(abs(expected_stat)))
    assert np.isclose(result["p_value"], expected_p)

    z = _norm.ppf(0.5 + level / 2.0)
    expected_lower = max(0.0, centre - z * se)
    expected_upper = min(1.0, centre + z * se)
    assert np.isclose(result["lower"], expected_lower)
    assert np.isclose(result["upper"], expected_upper)
    assert 0.0 <= result["lower"] <= 1.0
    assert 0.0 <= result["upper"] <= 1.0


def test_fzt53_no_null():
    """When null is None, statistic and p_value must be NaN; CI is still valid."""
    result = fauzi_thm5_3_bdfree_normality(0.7, 0.01, null=None, level=0.9)
    assert isinstance(result, dict)
    assert np.isnan(result["statistic"])
    assert np.isnan(result["p_value"])
    assert np.isclose(result["se"], np.sqrt(0.01))
    assert np.isclose(result["level"], 0.9)
    assert 0.0 <= result["lower"] <= 1.0
    assert 0.0 <= result["upper"] <= 1.0


def test_fzt53_edge():
    """Test edge cases: clipping the interval to [0, 1]."""
    result = fauzi_thm5_3_bdfree_normality(0.99, 0.25, null=0.5, level=0.95)
    assert isinstance(result, dict)
    assert result["lower"] >= 0.0
    assert result["upper"] <= 1.0
    assert np.isclose(result["se"], np.sqrt(0.25))

    result_low = fauzi_thm5_3_bdfree_normality(0.01, 0.25, null=0.5, level=0.95)
    assert result_low["lower"] >= 0.0
    assert result_low["upper"] <= 1.0
