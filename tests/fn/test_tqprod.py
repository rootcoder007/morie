"""Tests for tqprod.turboquant_qjl_product_estimator."""

from morie.fn import _array_core as np

from morie.fn.tqprod import turboquant_qjl_product_estimator


def test_tqprod_basic():
    """Test basic functionality."""
    q = np.ones(4)
    signs_k = [[1, 1, 1, 1]]
    norm_k = 1.0
    S = np.eye(4)
    result = turboquant_qjl_product_estimator(q, signs_k, norm_k, S)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_tqprod_edge():
    """Test edge cases."""
    q = np.ones(4)
    signs_k = [[1, 1, 1, 1]]
    norm_k = 1.0
    S = np.eye(4)
    result = turboquant_qjl_product_estimator(q, signs_k, norm_k, S)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.tqprod as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
