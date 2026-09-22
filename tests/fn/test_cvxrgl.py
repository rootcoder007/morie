"""Tests for cvxrgl.boyd_regularized_ls."""

from morie.fn import _array_core as np

from morie.fn.cvxrgl import boyd_regularized_ls


def test_cvxrgl_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    A = rng.normal(0, 1, (10, 3))
    b = rng.normal(0, 1, 10)
    delta = 0.5
    result = boyd_regularized_ls(A, b, delta)
    assert isinstance(result, dict)
    assert "x" in result
    assert "rss" in result
    assert "penalty" in result
    assert "objective" in result
    assert "effective_df" in result
    assert "shrinkage" in result
    x = result["x"]
    rss = result["rss"]
    penalty = result["penalty"]
    objective = result["objective"]
    expected_x = np.linalg.lstsq(
        np.vstack([A, np.sqrt(delta) * np.eye(3)]),
        np.r_[b, np.zeros(3)],
        rcond=None,
    )[0]
    assert np.allclose(x, expected_x)
    expected_rss = float((b - A @ x) @ (b - A @ x))
    expected_penalty = float(delta * (x @ x))
    assert rss == expected_rss
    assert penalty == expected_penalty
    assert objective == expected_rss + expected_penalty
    assert result["delta"] == delta
    assert result["method"] == "boyd_regularized_ls"
    assert result["shrinkage"] == float(np.linalg.norm(x))


def test_cvxrgl_edge():
    """Test edge cases."""
    A = np.array([[1.0, 1.0], [1.0, 1.0]])
    b = np.array([2.0, 2.0])
    r_small = boyd_regularized_ls(A, b, delta=0.1)
    r_large = boyd_regularized_ls(A, b, delta=10.0)
    assert isinstance(r_small, dict)
    assert isinstance(r_large, dict)
    assert r_small["shrinkage"] > r_large["shrinkage"]
