"""Tests for gb431.gibbons_ks_dist_free."""

from morie.fn import _array_core as np

from morie.fn.gb431 import gibbons_ks_dist_free


def _normal_cdf(x):
    """Standard normal CDF F_0 used as the hypothesised distribution."""
    return 0.5 * (1.0 + np.vectorize(_math_erf)(x / np.sqrt(2.0)))


def _math_erf(x):
    """erf math helper using math.erf via a tiny indirection."""
    import math
    return math.erf(x)


def test_gb431_basic():
    """Test basic functionality against the documented formula."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    result = gibbons_ks_dist_free(x, _normal_cdf)

    n = len(x)
    xs = sorted(float(v) for v in x)
    z = [_normal_cdf(v) for v in xs]

    # Formula from the docstring (Gibbons Thm 4.3.1).
    expected_dp = max((i + 1) / n - z[i] for i in range(n))
    expected_dm = max(z[i] - i / n for i in range(n))
    expected_stat = max(expected_dp, expected_dm)

    assert isinstance(result, dict)
    # Documented return keys.
    assert "statistic" in result
    assert "dplus" in result
    assert "dminus" in result
    assert "z" in result
    assert "n" in result
    assert "method" in result

    # Independent arithmetic check of the formula.
    assert result["statistic"] == float(expected_stat)
    assert result["dplus"] == float(expected_dp)
    assert result["dminus"] == float(expected_dm)
    assert result["n"] == n
    # PIT values are sorted Uniform(0,1) order statistics under H0.
    assert list(result["z"]) == z
    assert all(0.0 <= zv <= 1.0 for zv in result["z"])


def test_gb431_edge():
    """Test edge case: n = 1 sample should still work."""
    x = [0.0]
    result = gibbons_ks_dist_free(x, _normal_cdf)

    n = 1
    xs = sorted(float(v) for v in x)
    z = [_normal_cdf(v) for v in xs]

    expected_dp = max((i + 1) / n - z[i] for i in range(n))
    expected_dm = max(z[i] - i / n for i in range(n))
    expected_stat = max(expected_dp, expected_dm)

    assert isinstance(result, dict)
    assert result["statistic"] == float(expected_stat)
    assert result["dplus"] == float(expected_dp)
    assert result["dminus"] == float(expected_dm)
    assert result["n"] == n
    assert list(result["z"]) == z
