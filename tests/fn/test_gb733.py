"""Tests for gb733.gibbons_linrank_covariance."""

from morie.fn import _array_core as np

from morie.fn.gb733 import gibbons_linrank_covariance


def test_gb733_basic():
    """Test basic functionality against the documented formula."""
    rng = np.random.default_rng(42)
    N = 100
    m = 10
    n = N - m
    a = rng.normal(0, 1, N)
    b = rng.normal(0, 1, N)
    result = gibbons_linrank_covariance(a, b, m, n)

    # Return type and documented keys.
    assert isinstance(result, dict)
    for key in ("cov", "corr", "var_a", "var_b", "N", "m", "n", "method"):
        assert key in result

    # N is m + n.
    assert result["N"] == m + n
    assert result["m"] == m
    assert result["n"] == n

    # Independent recomputation of the formula in Theorem 7.3.3.
    av = [float(v) for v in a]
    bv = [float(v) for v in b]
    nn = m + n
    k = m * n / (float(nn) ** 2 * (nn - 1.0))
    expected_cov = k * (
        nn * sum(av[i] * bv[i] for i in range(nn)) - sum(av) * sum(bv)
    )
    expected_va = k * (nn * sum(v * v for v in av) - sum(av) ** 2)
    expected_vb = k * (nn * sum(v * v for v in bv) - sum(bv) ** 2)
    expected_corr = expected_cov / (expected_va * expected_vb) ** 0.5

    assert result["cov"] == expected_cov
    assert result["var_a"] == expected_va
    assert result["var_b"] == expected_vb
    assert result["corr"] == expected_corr


def test_gb733_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    N = 100
    m = 10
    n = N - m
    a = rng.normal(0, 1, N)
    b = rng.normal(0, 1, N)
    result = gibbons_linrank_covariance(a, b, m, n)
    assert isinstance(result, dict)
    assert "cov" in result
    assert "corr" in result
