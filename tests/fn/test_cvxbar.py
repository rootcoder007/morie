"""Tests for cvxbar.boyd_log_barrier."""

from morie.fn import _array_core as np

from morie.fn.cvxbar import boyd_log_barrier


def test_cvxbar_basic():
    """Test basic functionality."""
    f = np.array([-1.0, -2.0])
    result = boyd_log_barrier(f, t=10.0, f0=5.0)
    assert isinstance(result, dict) or hasattr(result, "__getitem__")
    expected_barrier = -float(np.sum(np.log(-np.asarray(f, dtype=float))))
    assert round(result["barrier"], 6) == round(expected_barrier, 6)
    assert round(result["barrier"], 6) == -0.693147
    assert result["centering_objective"] == 10.0 * 5.0 + expected_barrier
    assert round(result["gradient_factor"][0], 6) == 1.0
    assert round(result["gradient_factor"][1], 6) == 0.5
    assert result["suboptimality_bound"] == 2 / 10.0
    assert result["m"] == 2
    assert result["objective"] == 5.0


def test_cvxbar_edge():
    """Test edge cases."""
    f = np.array([-1.0] * 10)
    result = boyd_log_barrier(f, t=100.0)
    assert isinstance(result, dict) or hasattr(result, "__getitem__")
    assert result["suboptimality_bound"] == 10 / 100.0
    assert result["centering_objective"] is None
    assert result["m"] == 10
    barrier = -float(np.sum(np.log(-np.asarray(f, dtype=float))))
    assert round(result["barrier"], 6) == round(barrier, 6)
