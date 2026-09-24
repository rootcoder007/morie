"""Tests for hmlstm.geron_lstm."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.hmlstm import geron_lstm


_KEYS = (
    "W_i", "U_i", "b_i",
    "W_f", "U_f", "b_f",
    "W_o", "U_o", "b_o",
    "W_g", "U_g", "b_g",
)


def _make_weights(rng, n_in, n_units):
    """Build a weights mapping with the keys and shapes geron_lstm requires."""
    out = {}
    for k in _KEYS:
        if k.startswith("W"):
            out[k] = rng.normal(0, 0.1, (n_units, n_in))
        elif k.startswith("U"):
            out[k] = rng.normal(0, 0.1, (n_units, n_units))
        else:  # b_*
            out[k] = rng.normal(0, 0.1, (n_units,))
    return out


def test_hmlstm_basic():
    """One LSTM step on randomly initialised gates with matching input/hidden sizes."""
    n_in, n_units = 3, 2
    rng = np.random.default_rng(42)
    x_t = rng.normal(0, 1, (n_in,))
    h_prev = rng.normal(0, 1, (n_units,))
    c_prev = rng.normal(0, 1, (n_units,))
    weights = _make_weights(np.random.default_rng(45), n_in, n_units)
    result = geron_lstm(x_t, h_prev, c_prev, weights)
    assert isinstance(result, dict) or hasattr(result, "__getitem__")
    for key in ("h_t", "c_t", "i_t", "f_t", "o_t", "g_t", "estimate", "n", "method"):
        assert key in result
    for key in ("h_t", "c_t", "i_t", "f_t", "o_t", "g_t"):
        vals = list(result[key])
        assert len(vals) == n_units
        for v in vals:
            assert math.isfinite(float(v))
    for key in ("estimate", "n"):
        assert math.isfinite(float(result[key]))


def test_hmlstm_edge():
    """Empty x_t / h_prev are rejected; a valid small call returns the documented keys."""
    n_in, n_units = 3, 2
    rng = np.random.default_rng(42)
    x_t = rng.normal(0, 1, (n_in,))
    h_prev = rng.normal(0, 1, (n_units,))
    c_prev = rng.normal(0, 1, (n_units,))
    weights = _make_weights(np.random.default_rng(45), n_in, n_units)

    with pytest.raises(ValueError):
        geron_lstm([], h_prev, c_prev, weights)
    with pytest.raises(ValueError):
        geron_lstm(x_t, [], c_prev, weights)

    result = geron_lstm(x_t, h_prev, c_prev, weights)
    assert "h_t" in result
    assert "c_t" in result
    assert "i_t" in result
    assert "f_t" in result
    assert "o_t" in result
    assert "g_t" in result
    assert "estimate" in result
    assert len(list(result["h_t"])) == n_units
    assert len(list(result["c_t"])) == n_units


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmlstm as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
