"""Tests for gb821c.gibbons_wrs_ci."""

from morie.fn import _array_core as np

from morie.fn.gb821c import gibbons_wrs_ci


def test_gb821c_basic():
    """Test basic functionality."""
    rng_x = np.random.default_rng(42)
    rng_y = np.random.default_rng(43)
    x = rng_x.normal(0, 1, 10)
    y = rng_y.normal(0.5, 1, 12)
    m = len(x)
    n = len(y)
    # From Table J for the Mann-Whitney / Wilcoxon rank-sum statistic
    # with m=10, n=12 at alpha=0.05 (two-sided, normal approximation).
    wcrit = 103.0
    result = gibbons_wrs_ci(x, y, wcrit)

    assert isinstance(result, dict)
    for key in ("lower", "upper", "k", "estimate", "ndiff", "m", "n", "method"):
        assert key in result

    # Independent recomputation from the documented formula.
    xs = [float(v) for v in x]
    ys = [float(v) for v in y]
    d = sorted(xi - yj for xi in xs for yj in ys)
    nd = len(d)
    k = int(round(wcrit - m * (m + 1) / 2.0))
    expected_lower = float(d[k])
    expected_upper = float(d[nd - 1 - k])
    mid = nd // 2
    expected_estimate = d[mid] if nd % 2 else (d[mid - 1] + d[mid]) / 2.0
    expected_ndiff = int(nd)

    assert result["lower"] == expected_lower
    assert result["upper"] == expected_upper
    assert result["estimate"] == expected_estimate
    assert result["k"] == k
    assert result["ndiff"] == expected_ndiff
    assert result["m"] == m
    assert result["n"] == n
    assert result["lower"] <= result["estimate"] <= result["upper"]


def test_gb821c_edge():
    """Test edge cases."""
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [2.0, 4.0, 6.0]
    m = len(x)
    n = len(y)
    # k must land in 0..mn-1 with nd = m*n = 15.
    # Choose wcrit so that k = wcrit - m(m+1)/2 is a valid small index.
    wcrit = float(m * (m + 1) / 2.0 + 2.0)  # k = 2
    result = gibbons_wrs_ci(x, y, wcrit)
    assert isinstance(result, dict)
    for key in ("lower", "upper", "k", "estimate", "ndiff", "m", "n", "method"):
        assert key in result
    assert result["k"] == 2
    assert result["ndiff"] == m * n
    assert result["m"] == m
    assert result["n"] == n
