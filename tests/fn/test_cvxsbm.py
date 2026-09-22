"""Tests for cvxsbm.boyd_subgrad_method."""

from morie.fn import _array_core as np

from morie.fn.cvxsbm import boyd_subgrad_method


def test_cvxsbm_basic():
    """Test basic functionality."""
    f = lambda z: float(np.abs(z[0])) + float(np.abs(z[1]))
    subgrad = lambda z: np.array([np.sign(z[0]), np.sign(z[1])])
    x0 = np.array([3.0, -2.0])
    t = 0.5
    result = boyd_subgrad_method(f, subgrad, x0, t=t, max_iter=200, rule="sqrt")
    assert hasattr(result, "payload")
    payload = result.payload
    assert "x_best" in payload
    assert "f_best" in payload
    assert "x_last" in payload
    assert "f_last" in payload
    assert "increased" in payload
    assert "f_path" in payload
    assert isinstance(payload["f_path"], np.ndarray)
    assert payload["f_path"].shape == (201,)
    assert payload["x_best"].shape == (2,)
    assert payload["x_last"].shape == (2,)
    assert isinstance(payload["f_best"], float)
    assert isinstance(payload["f_last"], float)
    assert isinstance(payload["increased"], int)
    assert payload["f_best"] <= 0.05
    assert payload["f_last"] >= payload["f_best"]
    assert payload["increased"] > 0


def test_cvxsbm_edge():
    """Test edge cases."""
    f = lambda z: float(z[0] ** 2) + float(z[1] ** 2)
    subgrad = lambda z: 2.0 * z
    x0 = np.array([5.0, -4.0])
    t = 0.1
    result = boyd_subgrad_method(f, subgrad, x0, t=t, max_iter=300, rule="inverse")
    assert hasattr(result, "payload")
    payload = result.payload
    assert "x_best" in payload
    assert "f_best" in payload
    assert "x_last" in payload
    assert "f_last" in payload
    assert "increased" in payload
    assert "f_path" in payload
    assert isinstance(payload["f_path"], np.ndarray)
    assert payload["f_path"].shape == (301,)
    assert payload["x_best"].shape == (2,)
    assert payload["x_last"].shape == (2,)
    initial_f = float(x0[0] ** 2 + x0[1] ** 2)
    assert payload["f_best"] <= initial_f
