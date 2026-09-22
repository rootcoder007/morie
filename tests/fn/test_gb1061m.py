"""Tests for gb1061m.gibbons_jt_moments."""

from morie.fn import _array_core as np

from morie.fn.gb1061m import gibbons_jt_moments


def test_gb1061m_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    ns = rng.integers(1, 30, size=5)
    result = gibbons_jt_moments(ns)

    nn = int(sum(ns))
    sum_sq = sum(int(v) ** 2 for v in ns)
    sum_sq_lin = sum(int(v) ** 2 * (2 * int(v) + 3) for v in ns)

    expected_mean = (nn ** 2 - sum_sq) / 4.0
    expected_pair = sum(
        int(ns[i]) * int(ns[j]) / 2.0
        for i in range(len(ns))
        for j in range(i + 1, len(ns))
    )
    expected_var = (
        nn ** 2 * (2 * nn + 3) - sum_sq_lin
    ) / 72.0
    expected_sd = expected_var ** 0.5

    assert result.mean == expected_mean
    assert result.mean_pairwise == expected_pair
    assert result.var == expected_var
    assert result.sd == expected_sd
    assert result.k == len(ns)
    assert result.n == nn
    assert result.method == "JT null moments, eqs. (10.6.2)-(10.6.3)"


def test_gb1061m_edge():
    """Test edge cases."""
    ns = [3, 5, 7]
    result = gibbons_jt_moments(ns)

    nn = sum(ns)
    expected_mean = (nn ** 2 - sum(v ** 2 for v in ns)) / 4.0
    expected_pair = sum(
        ns[i] * ns[j] / 2.0
        for i in range(len(ns))
        for j in range(i + 1, len(ns))
    )

    assert result.mean == expected_mean
    assert result.mean_pairwise == expected_pair
    assert result.k == 3
    assert result.n == nn
