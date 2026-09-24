"""Tests for hmrnn.geron_recurrent_neuron."""

import math

from morie.fn import _array_core as np

from morie.fn.hmrnn import geron_recurrent_neuron


def test_hmrnn_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n_in = 3
    n_units = 4
    x_t = rng.normal(0, 1, n_in)
    h_prev = rng.normal(0, 1, n_units)
    Wx = rng.normal(0, 1, (n_units, n_in))
    Wh = rng.normal(0, 1, (n_units, n_units))
    b = rng.normal(0, 1, n_units)
    result = geron_recurrent_neuron(x_t, h_prev, Wx, Wh, b)
    assert isinstance(result, dict)
    assert "h" in result
    assert "z" in result
    assert "jacobian_norm" in result
    assert "estimate" in result
    assert "n" in result
    assert "method" in result
    assert len(result["h"]) == n_units
    assert len(result["z"]) == n_units
    assert len(result["estimate"]) == n_units
    assert int(result["n"]) == n_units
    assert math.isfinite(float(result["jacobian_norm"]))
    # Default activation is tanh, so hidden state must be bounded in [-1, 1]
    for v in result["h"]:
        assert -1.0 <= float(v) <= 1.0


def test_hmrnn_edge():
    """Test edge case: smallest valid configuration (scalar input, single unit)."""
    x_t = [1.0]
    h_prev = [0.0]
    Wx = [[1.0]]
    Wh = [[1.0]]
    b = [0.0]
    result = geron_recurrent_neuron(x_t, h_prev, Wx, Wh, b)
    assert isinstance(result, dict)
    assert "h" in result
    assert "z" in result
    assert "jacobian_norm" in result
    assert "estimate" in result
    assert "n" in result
    assert "method" in result
    assert len(result["h"]) == 1
    assert int(result["n"]) == 1
    assert math.isfinite(float(result["jacobian_norm"]))
    # tanh(1.0) lies in (-1, 1)
    assert -1.0 < float(result["h"][0]) < 1.0


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmrnn as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
