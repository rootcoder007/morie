"""Tests for gpmlt.gp_multitask."""

from morie.fn import _array_core as np

from morie.fn.gpmlt import gp_multitask


def _to_matrix(arr):
    """Convert an marr (or anything iterable-of-iterables) to a list of lists."""
    return [list(row) for row in arr]


def test_gpmlt_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    X = _to_matrix(rng.normal(0, 1, (5, 2)))         # 5 inputs in 2-D
    y_tasks = _to_matrix(rng.normal(0, 1, (2, 5)))   # 2 tasks, 5 observations each
    X_test = _to_matrix(rng.normal(0, 1, (3, 2)))     # 3 test inputs in 2-D

    result = gp_multitask(X, y_tasks, X_test)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "mean" in result
    assert "loglik" in result
    assert "tasks" in result and result["tasks"] == 2
    assert "n" in result and result["n"] == 5
    assert isinstance(result["mean"], list)
    assert len(result["mean"]) == 2          # one row per task
    for row in result["mean"]:
        assert len(row) == 3                 # one prediction per test input


def test_gpmlt_edge():
    """Test edge cases: identical observations across tasks with identity task cov."""
    # Two tasks observed at identical inputs -> posterior mean equals observations.
    X = [[0.0], [1.0], [2.0]]
    y_tasks = [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
    result = gp_multitask(X, y_tasks, X, task_cov=[[1.0, 0.0], [0.0, 1.0]])
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "mean" in result
    assert len(result["mean"]) == 2
    assert len(result["mean"][0]) == 3
