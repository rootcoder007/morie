"""Tests for hmmxp.geron_max_pool (Geron ch. 12 max pooling)."""

from morie.fn import _array_core as np
from morie.fn.hmmxp import geron_max_pool


def _x():
    return [[[float((7 * i + 3 * j) % 11), -float((5 * i + j) % 9)] for j in range(5)]
            for i in range(5)]


def test_hmmxp_basic():
    """2x2 windows at stride 2 on a 5x5x2 map: floor((5 - 2) / 2) + 1 = 2
    per side; each output is the window maximum, per channel."""
    x = _x()
    result = geron_max_pool(np.array(x), 2)
    assert isinstance(result, dict)
    got = np.asarray(result["pooled"]).tolist()
    assert result["output_shape"] == (2, 2, 2)
    for i in range(2):
        for j in range(2):
            for c in range(2):
                assert got[i][j][c] == max(x[2 * i + u][2 * j + v][c]
                                           for u in range(2) for v in range(2))


def test_hmmxp_edge():
    """Overlapping 3x3 windows at stride 1 on one channel."""
    x = [[row[j][0] for j in range(5)] for row in _x()]
    r = geron_max_pool(np.array(x), 3, stride=1)
    exp = [[max(x[i + u][j + v] for u in range(3) for v in range(3)) for j in range(3)]
           for i in range(3)]
    assert np.asarray(r["pooled"]).tolist() == exp
    assert int(r["parameters"]) == 0


# --- appended: the module's own worked example as a gate -----------

import doctest as _doctest

import morie.fn.hmmxp as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
