"""Tests for cvxcen.boyd_central_path."""

from morie.fn import _array_core as np

from morie.fn.cvxcen import boyd_central_path


def test_cvxcen_basic():
    """Test basic functionality."""
    obj = lambda x: 0.5 * x[0] ** 2
    con = [lambda x: 1.0 - x[0]]
    t = [1.0, 10.0, 100.0]
    r = boyd_central_path(obj, con, t, x0=[2.0])
    assert "path" in r
    assert "objective" in r
    assert "gap" in r
    assert "slack" in r
    assert "t" in r
    expected_path = np.array([(1 + np.sqrt(1 + 4 / ti)) / 2 for ti in t])
    assert np.allclose(r["path"][:, 0], expected_path)
    assert np.all(r["objective"] - 0.5 <= r["gap"] + 1e-09)
    assert np.all(np.diff(r["objective"]) < 0)
    assert np.all(r["slack"] < 0)


def test_cvxcen_edge():
    """Test edge cases."""
    obj = lambda x: 0.5 * x[0] ** 2
    con = [lambda x: 1.0 - x[0]]
    t = [1e-06, 1e+04]
    two = boyd_central_path(obj, [con[0], lambda x: x[0] - 3.0], t, x0=[2.0])
    assert round(float(two["path"][0, 0]), 4) == 2.0
    assert round(float(two["path"][1, 0]), 4) == 1.0001
