"""Tests for hmfmap.geron_feature_map: phi(conv(x, K) + b)."""

import math

import pytest

from morie.fn.hmfmap import geron_feature_map


def _x():
    return [[float((3 * i + 2 * j) % 5) - 2.0 for j in range(5)] for i in range(5)]


def test_hmfmap_basic():
    """Stride 2 with zero padding 1 and tanh, recomputed from the padded
    input: out = floor((5 + 2 - 3) / 2) + 1 = 3 per side."""
    x, K, b = _x(), [[0.5, -1.0, 0.0], [1.0, 0.25, -0.5], [0.0, 0.5, 1.0]], 0.1
    result = geron_feature_map(x, K, b=b, activation="tanh", stride=2, padding=1)
    assert isinstance(result, dict)
    xp = [[0.0] * 7] + [[0.0] + r + [0.0] for r in x] + [[0.0] * 7]
    pre = [[b + sum(xp[2 * i + u][2 * j + v] * K[u][v] for u in range(3) for v in range(3))
            for j in range(3)] for i in range(3)]
    assert result["out_shape"] == (3, 3)
    fm = result["feature_map"]
    for i in range(3):
        for j in range(3):
            assert result["pre_activation"][i][j] == pytest.approx(pre[i][j], rel=0, abs=1e-12)
            assert fm[i][j] == pytest.approx(math.tanh(pre[i][j]), rel=0, abs=1e-12)


def test_hmfmap_edge():
    """ReLU clips the negative part; sparsity is the fraction of exact
    zeros in the map, counted here."""
    x, K = _x(), [[1.0, -1.0], [-1.0, 1.0]]
    r = geron_feature_map(x, K, b=0.0)
    pre = [[x[i][j] - x[i][j + 1] - x[i + 1][j] + x[i + 1][j + 1] for j in range(4)]
           for i in range(4)]
    relu = [[max(v, 0.0) for v in row] for row in pre]
    assert r["feature_map"] == relu
    zeros = sum(1 for row in relu for v in row if v == 0.0)
    assert r["sparsity"] == zeros / 16.0


# --- appended: the module's own worked example as a gate -----------

import doctest as _doctest

import morie.fn.hmfmap as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
