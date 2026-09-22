"""Tests for cvxlsq.boyd_least_squares."""

from morie.fn import _array_core as np

from morie.fn.cvxlsq import boyd_least_squares


def test_cvxlsq_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    A = rng.normal(0, 1, (10, 3))
    b = rng.normal(0, 1, 10)
    result = boyd_least_squares(A, b)
    assert isinstance(result, dict)
    assert "x" in result
    assert "residual" in result
    assert "rss" in result
    assert "rank" in result
    assert "condition_number" in result
    assert "underdetermined" in result
    x = result["x"]
    assert x.shape == (3,)
    # Normal equations: A^T A x = A^T b
    expected_x = np.linalg.solve(A.T @ A, A.T @ b)
    assert np.max(np.abs(x - expected_x)) < 1e-10
    # Residual orthogonal to column space
    assert np.max(np.abs(A.T @ result["residual"])) < 1e-10


def test_cvxlsq_edge():
    """Test edge cases."""
    A = np.array([[1.0, 1.0], [2.0, 2.0]])
    b = np.array([1.0, 2.0])
    result = boyd_least_squares(A, b)
    assert isinstance(result, dict)
    assert bool(result["underdetermined"]) is True
    assert int(result["rank"]) == 1
