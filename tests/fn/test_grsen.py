"""Tests for grsen.geron_senet_squeeze_excite."""

from morie.fn import _array_core as np

from morie.fn.grsen import geron_senet_squeeze_excite


def test_grsen_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    C = 3
    H, W = 4, 5
    X = rng.normal(0, 1, (H, W, C))
    W1 = rng.normal(0, 1, (C, C))
    W2 = rng.normal(0, 1, (C, C))
    result = geron_senet_squeeze_excite(X, W1, W2)
    assert isinstance(result, dict)
    assert "scale" in result
    assert "squeeze" in result
    assert "output" in result
    scale = list(result["scale"])
    assert len(scale) == C
    for s in scale:
        assert 0.0 < float(s) < 1.0
    squeeze = list(result["squeeze"])
    assert len(squeeze) == C


def test_grsen_edge():
    """Test edge cases."""
    X = [[[1.0, 3.0]]]
    I = [[1.0, 0.0], [0.0, 1.0]]
    result = geron_senet_squeeze_excite(X, I, I)
    assert isinstance(result, dict)
    assert "scale" in result
    assert "squeeze" in result
    assert "output" in result
    squeeze = list(result["squeeze"])
    assert len(squeeze) == 2
    assert abs(float(squeeze[0]) - 1.0) < 1e-6
    assert abs(float(squeeze[1]) - 3.0) < 1e-6


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grsen as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
