"""Tests for hmrvn.geron_revnet."""

import math

from morie.fn import _array_core as np

from morie.fn.hmrvn import geron_revnet


def test_hmrvn_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 8)
    F = lambda a: 2 * a
    G = lambda a: a + 1
    result = geron_revnet(x, F, G)
    assert isinstance(result, dict)
    for key in ("y", "y1", "y2", "x_reconstructed", "reconstruction_error"):
        assert key in result
    y = [float(v) for v in result["y"]]
    assert len(y) == 8
    x_rec = [float(v) for v in result["x_reconstructed"]]
    assert len(x_rec) == 8
    err = float(result["reconstruction_error"])
    assert math.isfinite(err)
    assert err < 1e-10


def test_hmrvn_edge():
    """Test edge cases."""
    x = [1.0, 2.0, 3.0, 4.0]
    F = lambda a: 2 * a
    G = lambda a: a + 1
    result = geron_revnet(x, F, G)
    assert isinstance(result, dict)
    assert [float(v) for v in result["y"]] == [7.0, 10.0, 11.0, 15.0]
    assert [float(v) for v in result["x_reconstructed"]] == [1.0, 2.0, 3.0, 4.0]
    assert float(result["reconstruction_error"]) == 0.0


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmrvn as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
