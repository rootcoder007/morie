"""Tests for hmphp.geron_peephole_lstm."""

import math

from morie.fn import _array_core as np

from morie.fn.hmphp import geron_peephole_lstm


def test_hmphp_basic():
    """Plain-LSTM path: no peephole vectors supplied, so they default to zeros."""
    rng = np.random.default_rng(42)
    n, H = 3, 2
    x_t = rng.normal(0, 1, n)
    h_prev = rng.normal(0, 1, H)
    c_prev = rng.normal(0, 1, H)
    weights = {
        "W_x": rng.normal(0, 1, (4 * H, n)),
        "W_h": rng.normal(0, 1, (4 * H, H)),
        "b": rng.normal(0, 1, 4 * H),
    }
    result = geron_peephole_lstm(x_t, h_prev, c_prev, weights)
    assert isinstance(result, dict)
    for key in ("h", "c", "i", "f", "g", "o", "estimate", "n", "method"):
        assert key in result
    assert len(result["h"]) == H
    assert len(result["c"]) == H
    for value in result["h"]:
        assert math.isfinite(float(value))
    for value in result["c"]:
        assert math.isfinite(float(value))


def test_hmphp_edge():
    """Peephole path on a minimal 1-unit LSTM with all three peephole vectors given."""
    rng = np.random.default_rng(7)
    n, H = 1, 1
    x_t = rng.normal(0, 1, n)
    h_prev = rng.normal(0, 1, H)
    c_prev = rng.normal(0, 1, H)
    weights = {
        "W_x": rng.normal(0, 1, (4 * H, n)),
        "W_h": rng.normal(0, 1, (4 * H, H)),
        "b": rng.normal(0, 1, 4 * H),
        "p_i": rng.normal(0, 1, H),
        "p_f": rng.normal(0, 1, H),
        "p_o": rng.normal(0, 1, H),
    }
    result = geron_peephole_lstm(x_t, h_prev, c_prev, weights)
    assert isinstance(result, dict)
    assert len(result["h"]) == H
    assert len(result["c"]) == H
    assert len(result["i"]) == H
    assert len(result["f"]) == H
    assert len(result["o"]) == H
    for key in ("i", "f", "o"):
        for value in result[key]:
            assert 0.0 <= float(value) <= 1.0


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmphp as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
