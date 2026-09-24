"""Tests for grqat.geron_quantization_aware_training."""

from morie.fn import _array_core as np

import math

from morie.fn.grqat import geron_quantization_aware_training


def test_grqat_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 40)
    s = 0.1
    bits = 8
    result = geron_quantization_aware_training(x, s, bits)
    assert isinstance(result, dict)
    assert "y" in result
    assert "ste_mask" in result
    assert "clipped_fraction" in result
    assert "grad_x" in result
    y = result["y"]
    assert len(y) == 40
    for v in y:
        assert math.isfinite(float(v))
    cf = float(result["clipped_fraction"])
    assert 0.0 <= cf <= 1.0
    for m in result["ste_mask"]:
        assert float(m) in (0.0, 1.0)
    for v in result["grad_x"]:
        assert math.isfinite(float(v))


def test_grqat_edge():
    """Test edge cases with an explicit upstream gradient."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 10)
    s = 0.5
    bits = 4
    g = rng.normal(0, 1, 10)
    result = geron_quantization_aware_training(x, s, bits, upstream_grad=g)
    assert isinstance(result, dict)
    assert "y" in result
    assert "grad_x" in result
    assert len(result["y"]) == 10
    assert len(result["grad_x"]) == 10
    for v in result["grad_x"]:
        assert math.isfinite(float(v))
    cf = float(result["clipped_fraction"])
    assert 0.0 <= cf <= 1.0


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grqat as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
