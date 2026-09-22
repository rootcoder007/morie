"""Tests for gb651.gibbons_ctrl_median."""

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

from morie.fn.gb651 import gibbons_ctrl_median


def test_gb651_basic():
    """Test basic functionality with a documented (odd-sized) control sample."""
    rng_x = np.random.default_rng(42)
    rng_y = np.random.default_rng(43)
    x = rng_x.normal(0, 1, 100)
    # Control sample y must have odd size per the function's docstring (n = 2r + 1).
    y = rng_y.normal(0, 1, 101)

    result = gibbons_ctrl_median(x, y)

    # The function returns a RichResult (mapping-like).
    assert isinstance(result, dict)
    for key in ("statistic", "p_value", "z", "mean", "var",
                "pmf", "r", "m", "n", "method"):
        assert key in result

    # Independently compute the statistic V from the documented definition:
    # V is the number of X observations preceding (i.e. <=) the Y median,
    # where the Y median is the middle element of the sorted odd-sized Y sample.
    xs_sorted = sorted(float(v) for v in x)
    ys_sorted = sorted(float(v) for v in y)
    m_expected = len(xs_sorted)
    n_expected = len(ys_sorted)
    r_expected = (n_expected - 1) // 2
    my_expected = ys_sorted[r_expected]
    v_expected = sum(1 for t in xs_sorted if t <= my_expected)
    mean_expected = m_expected / 2.0
    var_expected = m_expected * (m_expected + n_expected) / (4.0 * n_expected)

    assert n_expected % 2 == 1  # sanity: control sample is odd
    assert result["m"] == m_expected
    assert result["n"] == n_expected
    assert result["r"] == r_expected
    assert result["statistic"] == v_expected
    assert result["mean"] == mean_expected
    assert result["var"] == var_expected

    # Independently compute Z from the asymptotic formula in the docstring.
    z_expected = (v_expected - m_expected / 2.0) / (var_expected ** 0.5)
    assert abs(result["z"] - z_expected) < 1e-12

    # pmf must sum to 1 (it's a probability mass function).
    assert abs(sum(result["pmf"]) - 1.0) < 1e-12
    assert len(result["pmf"]) == m_expected + 1

    # p_value is a probability in [0, 1].
    assert 0.0 <= result["p_value"] <= 1.0


def test_gb651_edge():
    """Test edge case: a small odd-sized control sample."""
    rng_x = np.random.default_rng(42)
    rng_y = np.random.default_rng(43)
    x = rng_x.normal(0, 1, 10)
    # Smallest non-trivial odd size for y is 3.
    y = rng_y.normal(0, 1, 3)

    result = gibbons_ctrl_median(x, y)

    assert isinstance(result, dict)
    assert result["m"] == 10
    assert result["n"] == 3
    assert result["r"] == 1
    assert len(result["pmf"]) == 11  # m + 1 entries
    assert abs(sum(result["pmf"]) - 1.0) < 1e-12
    assert 0.0 <= result["p_value"] <= 1.0
