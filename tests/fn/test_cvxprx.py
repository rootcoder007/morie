"""Tests for cvxprx.boyd_proximal."""

from morie.fn import _array_core as np

from morie.fn.cvxprx import boyd_proximal


def test_cvxprx_basic():
    """Test basic functionality."""
    h = "l1"
    rng = np.random.default_rng(44)
    v = rng.normal(0, 1, 100)
    t = 0.3
    result = boyd_proximal(h, v, t=t)
    assert isinstance(result, dict)
    assert "prox" in result
    assert "moreau" in result
    assert "h_name" in result
    assert result["h_name"] == "l1"
    assert result["t"] == float(t)

    prox = np.asarray(result["prox"], dtype=float)
    v_arr = np.asarray(v, dtype=float)
    expected = np.sign(v_arr) * np.maximum(np.abs(v_arr) - t, 0.0)
    assert prox.shape == v_arr.shape
    assert np.allclose(prox, expected)

    zeroed = np.abs(v_arr) <= t
    assert np.allclose(prox[zeroed], 0.0)
    assert np.allclose(prox[~zeroed], np.sign(v_arr[~zeroed]) * (np.abs(v_arr[~zeroed]) - t))


def test_cvxprx_edge():
    """Test edge cases."""
    h = "nonneg"
    v = np.array([-2.0, -0.5, 0.0, 1.5, 3.0])
    result = boyd_proximal(h, v)
    assert isinstance(result, dict)
    assert "prox" in result
    assert result["h_name"] == "nonneg"
    prox = np.asarray(result["prox"], dtype=float)
    expected = np.maximum(v, 0.0)
    assert np.allclose(prox, expected)
    assert np.all(prox >= 0.0)
