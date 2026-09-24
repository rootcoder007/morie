"""Tests for hmmcel.geron_memory_cell."""
import math

import pytest

from morie.fn import _array_core as np

from morie.fn.hmmcel import geron_memory_cell


def test_hmmcel_basic():
    """Test basic single-step functionality with the leaky integrator."""
    leaky = lambda c, x: 0.5 * np.asarray(c, dtype=float) + np.asarray(x, dtype=float)
    rng = np.random.default_rng(42)
    n_units = 4
    c_prev = rng.normal(0, 1, n_units)
    x_t = rng.normal(0, 1, n_units)
    result = geron_memory_cell(c_prev, x_t, leaky)
    assert isinstance(result, dict)
    for key in ("c_t", "states", "deltas", "n_steps", "estimate", "n", "method"):
        assert key in result
    assert result["n_steps"] == 1
    c_t = [float(v) for v in result["c_t"]]
    assert len(c_t) == n_units
    assert all(math.isfinite(v) for v in c_t)
    # The leaky cell: c_t[i] = 0.5 * c_prev[i] + x_t[i].
    expected = [0.5 * float(c) + float(x) for c, x in zip(c_prev, x_t)]
    for actual, exp in zip(c_t, expected):
        assert math.isclose(actual, exp, rel_tol=1e-9, abs_tol=1e-12)


def test_hmmcel_edge():
    """Test unrolled 2-D sequence and shape-change rejection."""
    leaky = lambda c, x: 0.5 * np.asarray(c, dtype=float) + np.asarray(x, dtype=float)
    rng = np.random.default_rng(0)
    T = 5
    n_in = 3
    x_seq = rng.normal(0, 1, (T, n_in))
    c0 = rng.normal(0, 1, n_in)
    result = geron_memory_cell(c0, x_seq, leaky)
    assert isinstance(result, dict)
    assert result["n_steps"] == T
    states = result["states"]
    assert len(states) == T
    deltas = [float(d) for d in result["deltas"]]
    assert len(deltas) == T
    assert all(math.isfinite(v) for v in deltas)
    # The function's documented contract: f must preserve c's shape.
    with pytest.raises(ValueError):
        geron_memory_cell([0.0, 0.0], [1.0, 1.0], lambda c, x: np.array([0.0]))


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmmcel as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
