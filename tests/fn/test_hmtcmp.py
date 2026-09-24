"""Tests for hmtcmp.geron_torch_compile."""

from morie.fn import _array_core as np

from morie.fn.hmtcmp import geron_torch_compile


def test_hmtcmp_basic():
    """Test basic functionality."""
    A = np.eye(2) * 2
    B = np.eye(2) * 3
    C = np.eye(2) * 5
    x = [[1.0, 1.0]]
    result = geron_torch_compile(
        [("linear", A), ("linear", B), ("linear", C)],
        example_inputs=x,
    )
    assert isinstance(result, dict)
    assert "n_ops" in result
    assert "n_compiled" in result
    assert "output" in result
    assert "max_diff" in result
    assert result["n_ops"] == 3
    assert result["n_compiled"] == 1
    assert result["output"][0][0] == 30.0
    assert result["output"][0][1] == 30.0
    assert result["max_diff"] < 1e-12


def test_hmtcmp_edge():
    """Test edge cases."""
    A = np.eye(2) * 2
    B = np.eye(2) * 3
    x = [[1.0, 1.0]]
    result = geron_torch_compile(
        [("linear", A), ("relu",), ("linear", B)],
        example_inputs=x,
    )
    assert isinstance(result, dict)
    assert result["n_compiled"] == 3


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmtcmp as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
