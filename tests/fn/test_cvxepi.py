"""Tests for cvxepi.boyd_epigraph."""

from morie.fn import _array_core as np

from morie.fn.cvxepi import boyd_epigraph


def test_cvxepi_basic():
    """Test basic functionality."""
    f = lambda x: x ** 2
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    result = boyd_epigraph(f, x, t)
    assert isinstance(result, dict)
    assert "in_epigraph" in result
    assert "fx" in result
    assert "slack" in result
    assert "on_boundary" in result
    assert "fraction_inside" in result

    x_arr = np.asarray(x, dtype=float).ravel()
    t_arr = np.asarray(t, dtype=float).ravel()
    fx_expected = np.asarray([float(f(v)) for v in x_arr], dtype=float)
    slack_expected = t_arr - fx_expected
    inside_expected = slack_expected >= 0

    assert np.array_equal(result["fx"], fx_expected)
    assert np.array_equal(result["slack"], slack_expected)
    assert np.array_equal(result["in_epigraph"], inside_expected)
    assert result["fraction_inside"] == float(inside_expected.mean())


def test_cvxepi_edge():
    """Test edge cases."""
    f = lambda x: x ** 2
    b = boyd_epigraph(f, [2.0], [4.0])
    assert isinstance(b, dict)
    assert bool(b["on_boundary"][0]) is True
    assert float(b["slack"][0]) == 0.0

    r = boyd_epigraph(f, [1.0, 1.0], [2.0, 0.5])
    assert isinstance(r, dict)
    assert [bool(v) for v in r["in_epigraph"]] == [True, False]
