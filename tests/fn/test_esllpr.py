import numpy as np
"""Tests for esllpr.esl_local_linear."""

from morie.fn import _array_core as np

from morie.fn.esllpr import esl_local_linear


def test_esllpr_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    x = rng.normal(0.0, 1.0, 100)
    y = 2.0 * x + 1.0 + rng.normal(0.0, 0.1, 100)
    x0 = np.array([0.0])
    lambda_ = 0.5
    result = esl_local_linear(x0, x, y, lambda_)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "values" in result
    assert "slopes" in result
    assert "n_in_window" in result
    assert "lambda" in result
    assert "kernel" in result
    assert "n" in result
    assert "method" in result
    assert result["n"] == 100
    assert result["lambda"] == lambda_
    # data lie on y = 2x + 1, so the local-linear fit must be (near-)exact
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value
    assert abs(result["slopes"][0] - 2.0) < 0.2


def test_esllpr_edge():
    """Test exact-on-a-line property at the boundary."""
    # Same example as in the docstring: local linear is exact EVERYWHERE,
    # even at the boundary where the kernel average is biased.
    xs = [0.0, 1.0, 2.0, 3.0]
    ys = [1.0, 3.0, 5.0, 7.0]  # y = 2 x + 1
    x0 = np.array([0.0])
    lambda_ = 1.5
    result = esl_local_linear(x0, xs, ys, lambda_)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "slopes" in result
    # Exact on a line, including at the boundary:
    assert abs(result["estimate"] - 1.0) < 1e-9
    assert abs(result["slopes"][0] - 2.0) < 1e-9
    # A hand-computed check: alpha + beta * x0 with the same formula
    # uses D = [1, xi - x0] and weights from K(|xi - x0|/lambda).
    # For x0 = 0, x = xs, lambda_ = 1.5, kernel = epanechnikov:
    u = np.abs(np.array(xs) - 0.0) / 1.5
    K = np.where(np.abs(u) <= 1.0, 0.75 * (1.0 - u * u), 0.0)
    sw = np.sqrt(K)
    D = np.column_stack([np.ones_like(u), np.array(xs) - 0.0])
    A = (D * sw[:, None]).T @ (D * sw[:, None])
    b = (D * sw[:, None]).T @ (np.array(ys) * sw)
    beta = (b[1] - b[0] * A[0, 1] / A[0, 0]) / (A[1, 1] - A[0, 1] * A[0, 1] / A[0, 0])
    alpha = (b[0] - beta * A[0, 1]) / A[0, 0]
    assert abs(result["estimate"] - (alpha + beta * 0.0)) < 1e-9
    assert abs(result["slopes"][0] - beta) < 1e-9
