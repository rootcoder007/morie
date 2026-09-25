"""Tests for hmfcn.geron_fcn (fully convolutional network forward pass)."""

import pytest

from morie.fn import _array_core as np
from morie.fn.hmfcn import geron_fcn


def _xcorr(x, k, b):
    # x (C, H, W), k (F, C, kh, kw): valid cross-correlation plus bias
    C, H, W = len(x), len(x[0]), len(x[0][0])
    F, kh, kw = len(k), len(k[0][0]), len(k[0][0][0])
    return [[[b[f] + sum(x[c][i + u][j + v] * k[f][c][u][v]
                         for c in range(C) for u in range(kh) for v in range(kw))
              for j in range(W - kw + 1)] for i in range(H - kh + 1)]
            for f in range(F)]


def _img():
    return [[[float((i * 5 + j * 3 + c) % 7) - 3.0 for j in range(5)]
             for i in range(4)] for c in range(2)]


def _k(F, C, s):
    return [[[[float((f + 2 * c + u - v + s) % 5) - 2.0 for v in range(2)]
              for u in range(2)] for c in range(C)] for f in range(F)]


def test_hmfcn_basic():
    """Two layers, ReLU between them: the class scores equal the
    cross-correlations recomputed here, and the segmentation is their
    per-pixel argmax."""
    x = _img()
    k1, b1 = _k(3, 2, 0), [0.5, -1.0, 0.0]
    k2, b2 = _k(2, 3, 1), [0.0, 0.25]
    result = geron_fcn(x, [(np.array(k1), b1, 1), (np.array(k2), b2, 1)])
    assert isinstance(result, dict)
    h1 = [[[max(v, 0.0) for v in row] for row in fm] for fm in _xcorr(x, k1, b1)]
    s = _xcorr(h1, k2, b2)
    got = np.asarray(result["class_map"]).tolist()
    assert result["out_shape"] == (2, 2, 3)
    for f in range(2):
        for i in range(2):
            for j in range(3):
                assert got[f][i][j] == pytest.approx(s[f][i][j], rel=1e-12, abs=1e-12)
    seg = [[max(range(2), key=lambda f: s[f][i][j]) for j in range(3)] for i in range(2)]
    assert result["segmentation"] == seg


def test_hmfcn_edge():
    """Nearest-neighbour upsampling by 2 repeats each class label over a
    2x2 block; a 1-D image is refused."""
    x = _img()
    k1, b1 = _k(2, 2, 3), [0.0, 0.0]
    r1 = geron_fcn(x, [(np.array(k1), b1, 1)])
    r2 = geron_fcn(x, [(np.array(k1), b1, 1)], upsample=2)
    seg = r1["segmentation"]
    up = [[seg[i // 2][j // 2] for j in range(2 * len(seg[0]))] for i in range(2 * len(seg))]
    assert r2["segmentation"] == up
    with pytest.raises(ValueError):
        geron_fcn([1.0, 2.0, 3.0], [np.array(k1)])


# --- appended: the module's own worked example as a gate -----------

import doctest as _doctest

import morie.fn.hmfcn as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
