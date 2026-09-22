"""Tests for gb_cvmc.gibbons_cramer_von_mises."""

from morie.fn import _array_core as np

from morie.fn.gb_cvmc import gibbons_cramer_von_mises


def _normal_cdf(z):
    """Standard normal CDF via the math library error function."""
    import math
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def test_gb_cvmc_basic():
    """Test basic functionality against the documented computing formula."""
    rng = np.random.default_rng(42)
    x = list(rng.normal(0, 1, 100))

    result = gibbons_cramer_von_mises(x, _normal_cdf)

    assert isinstance(result, dict)
    assert "statistic" in result
    assert "nw2" in result
    assert "z" in result
    assert "n" in result
    assert "method" in result

    xs = sorted(float(v) for v in x)
    n = len(xs)
    z = [_normal_cdf(v) for v in xs]
    expected_w2 = 1.0 / (12.0 * n) + sum(
        (z[j] - (2.0 * (j + 1) - 1.0) / (2.0 * n)) ** 2 for j in range(n)
    )

    assert result["n"] == n
    assert result["statistic"] == float(expected_w2)
    assert result["nw2"] == float(n * expected_w2)
    assert result["z"] == z


def test_gb_cvmc_edge():
    """Test edge cases: single sample, and the function-as-callable contract."""
    rng = np.random.default_rng(42)
    x = list(rng.normal(0, 1, 100))

    result = gibbons_cramer_von_mises(x, _normal_cdf)
    assert isinstance(result, dict)
    assert "statistic" in result
    assert result["n"] == len(x)

    # Single sample: still well-defined (n >= 1 is the only constraint).
    result_single = gibbons_cramer_von_mises([0.0], _normal_cdf)
    assert result_single["n"] == 1
    p = _normal_cdf(0.0)
    expected_single = 1.0 / 12.0 + (p - 0.5) ** 2
    assert result_single["statistic"] == float(expected_single)
