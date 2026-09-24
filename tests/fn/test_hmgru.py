"""Tests for hmgru.geron_gru."""

from morie.fn import _array_core as np

from morie.fn.hmgru import geron_gru


def _make_weights(rng, n_in, n_units):
    weights = {}
    for key in ("W_z", "U_z", "W_r", "U_r", "W_h", "U_h"):
        if key.startswith("W"):
            weights[key] = rng.normal(0, 1, (n_units, n_in))
        else:
            weights[key] = rng.normal(0, 1, (n_units, n_units))
    for key in ("b_z", "b_r", "b_h"):
        weights[key] = rng.normal(0, 1, n_units)
    return weights


def test_hmgru_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n_in, n_units = 3, 4
    x_t = rng.normal(0, 1, n_in)
    h_prev = rng.normal(0, 1, n_units)
    weights = _make_weights(rng, n_in, n_units)
    result = geron_gru(x_t, h_prev, weights)
    assert isinstance(result, dict)
    assert "h_t" in result
    assert "z_t" in result
    assert "r_t" in result
    assert "h_tilde" in result
    assert "estimate" in result
    assert "n" in result
    assert "method" in result
    assert len(result["h_t"]) == n_units
    assert len(result["z_t"]) == n_units
    assert len(result["r_t"]) == n_units
    assert len(result["h_tilde"]) == n_units


def test_hmgru_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n_in, n_units = 2, 2
    x_t = rng.normal(0, 1, n_in)
    h_prev = rng.normal(0, 1, n_units)
    weights = _make_weights(rng, n_in, n_units)
    result = geron_gru(x_t, h_prev, weights)
    assert isinstance(result, dict)
    assert "h_t" in result
    assert len(result["h_t"]) == n_units


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmgru as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
