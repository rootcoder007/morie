"""Tests for greast.geron_early_stopping."""

from morie.fn import _array_core as np

from morie.fn.greast import geron_early_stopping


def test_greast_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n, p = 40, 3
    X_train = rng.normal(0, 1, (n, p))
    y_train = rng.normal(0, 1, n)
    X_val = rng.normal(0, 1, (n, p))
    y_val = rng.normal(0, 1, n)
    n_iter = 50
    eta = 0.02
    result = geron_early_stopping(X_train, y_train, X_val, y_val, n_iter, eta)
    assert isinstance(result, dict)
    for key in ("theta", "best_iteration", "best_val_rmse",
                "val_rmse_history", "train_rmse_history",
                "final_val_rmse", "overfitting_detected"):
        assert key in result
    assert 0 <= result["best_iteration"] <= n_iter
    assert len(result["val_rmse_history"]) == n_iter + 1
    assert len(result["train_rmse_history"]) == n_iter + 1


def test_greast_edge():
    """Test edge cases."""
    # Clean linear data from the docstring example: validation error
    # falls the whole way, so the best snapshot is the last one.
    X_train = [[1.0, float(i)] for i in range(6)]
    y_train = [float(i) for i in range(6)]
    X_val = [[1.0, 10.0], [1.0, 11.0]]
    y_val = [10.0, 11.0]
    n_iter = 50
    eta = 0.02
    result = geron_early_stopping(X_train, y_train, X_val, y_val, n_iter, eta)
    assert isinstance(result, dict)
    assert result["best_iteration"] == 50
    assert result["best_val_rmse"] < result["val_rmse_history"][0]


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.greast as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
