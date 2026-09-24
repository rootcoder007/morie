"""Tests for grhbb.geron_hebb_rule."""

from morie.fn import _array_core as np

from morie.fn.grhbb import geron_hebb_rule


def test_grhbb_basic():
    """Test basic functionality on a small multi-input multi-output instance."""
    rng = np.random.default_rng(42)
    n_in, n_out = 5, 3
    x = rng.normal(0, 1, n_in)
    y_true = [float(v) for v in rng.integers(0, 2, n_out)]
    y_pred = rng.normal(0, 1, n_out)
    w = rng.normal(0, 1, (n_in, n_out))
    eta = 0.1
    result = geron_hebb_rule(x, y_true, y_pred, w, eta)
    assert isinstance(result, dict)
    expected_keys = {"w_new", "delta_w", "error", "converged",
                     "update_norm", "estimate", "n", "method"}
    assert expected_keys.issubset(set(result.keys()))
    assert isinstance(result["converged"], bool)


def test_grhbb_edge():
    """Test edge case: perfect prediction freezes the weights (converged=True)."""
    x = [1.0, 2.0]
    y_true = [1.0]
    y_pred = [1.0]
    w = [[0.1], [0.2]]
    eta = 0.1
    result = geron_hebb_rule(x, y_true, y_pred, w, eta)
    assert isinstance(result, dict)
    assert result["converged"] is True


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grhbb as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
