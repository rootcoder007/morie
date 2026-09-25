"""Tests for hmswin.geron_swin (windowed and shifted-window attention)."""

from morie.fn import _array_core as np
from morie.fn.hmswin import geron_swin


def _img():
    return [[float((3 * i + 5 * j) % 7) for j in range(4)] for i in range(4)]


def _changed(a, b):
    A, B = np.asarray(a).tolist(), np.asarray(b).tolist()
    return {(i, j) for i in range(4) for j in range(4)
            if any(abs(x - y) > 1e-12 for x, y in zip(A[i][j], B[i][j]))}


def test_hmswin_basic():
    """Without a shifted block attention stays inside each 2x2 window:
    perturbing pixel (0, 0) changes exactly the four tokens of the
    top-left window."""
    img = _img()
    result = geron_swin(img, window_size=2, n_layers=1)
    assert isinstance(result, dict)
    img2 = [row[:] for row in img]
    img2[0][0] += 1.0
    r2 = geron_swin(img2, window_size=2, n_layers=1)
    assert _changed(result["Y"], r2["Y"]) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert int(result["n_windows"]) == 4
    assert int(result["shifted_layers"]) == 0


def test_hmswin_edge():
    """The shifted second block carries the perturbation across the
    window boundary, which is the point of the shift."""
    img = _img()
    img2 = [row[:] for row in img]
    img2[1][1] += 1.0
    a = geron_swin(img, window_size=2, n_layers=2)
    b = geron_swin(img2, window_size=2, n_layers=2)
    ch = _changed(a["Y"], b["Y"])
    assert ch - {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert int(a["shifted_layers"]) == 1


# --- appended: the module's own worked example as a gate -----------

import doctest as _doctest

import morie.fn.hmswin as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
