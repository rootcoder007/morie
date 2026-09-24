"""Tests for jocros.joseph_croston_intermittent."""

import math

from morie.fn import _array_core as np

from morie.fn.jocros import joseph_croston_intermittent


def test_jocros_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(43)
    # Intermittent demand: many zeros with occasional non-zero values.
    y = rng.integers(0, 10, size=100)
    result = joseph_croston_intermittent(y, alpha=0.1)
    assert isinstance(result, dict)
    # Required output fields per docstring.
    assert "forecast" in result
    assert "classification" in result
    assert "cv_squared" in result
    assert "average_interval" in result
    # Numerical sanity checks without locking specific values.
    assert math.isfinite(result["forecast"])
    assert result["classification"] in {"smooth", "erratic", "intermittent", "lumpy"}
    assert math.isfinite(result["cv_squared"])
    assert result["cv_squared"] >= 0
    assert math.isfinite(result["average_interval"])


def test_jocros_edge():
    """Test edge cases."""
    rng = np.random.default_rng(44)
    # Constant positive demand: interval = 1, CV^2 = 0 -> "smooth".
    y = [5] * 20
    result = joseph_croston_intermittent(y, alpha=0.2)
    assert isinstance(result, dict)
    assert "forecast" in result
    assert "classification" in result
    assert "cv_squared" in result
    assert "average_interval" in result
    assert math.isfinite(result["forecast"])
    assert result["classification"] in {"smooth", "erratic", "intermittent", "lumpy"}
    assert math.isfinite(result["cv_squared"])
    assert result["cv_squared"] >= 0
    assert math.isfinite(result["average_interval"])


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.jocros as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
