"""Tests for gb45lf.gibbons_lilliefors_normal."""

from morie.fn import _array_core as np
from morie.fn.gb45lf import gibbons_lilliefors_normal


def test_gb45lf_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_lilliefors_normal(x)
    assert isinstance(result, dict)
    assert "statistic" in result or "p_value" in result or "estimate" in result


def test_gb45lf_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_lilliefors_normal(x)
    assert isinstance(result, dict)


def test_lilliefors_d_and_table_o_row_recomputed():
    import math

    import pytest

    x = [2.1, 3.4, 1.9, 5.6, 2.8, 3.1, 9.9, 2.5, 3.3, 2.7, 3.0, 2.4, 4.1]
    n = 13
    m = sum(x) / n
    s = math.sqrt(sum((v - m) ** 2 for v in x) / (n - 1))
    z = [0.5 * math.erfc(-((v - m) / s) / math.sqrt(2)) for v in sorted(x)]
    d = max(max((i + 1) / n - z[i], z[i] - i / n) for i in range(n))
    r = gibbons_lilliefors_normal(x, alpha=0.01)
    assert r["statistic"] == pytest.approx(d, rel=1e-12)
    assert r["n_table"] == 12 and r["dcrit"] == 0.281
    assert r["reject"] == int(d > 0.281)
