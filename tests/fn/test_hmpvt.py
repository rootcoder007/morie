"""Tests for hmpvt.geron_pvt (Pyramid Vision Transformer patch stages)."""

import pytest

from morie.fn import _array_core as np
from morie.fn.hmpvt import geron_pvt


def test_hmpvt_basic():
    """One stage, 2x2 patches of a 4x4x3 image: each token is the
    row-major flattened patch (u, v, c) times W, recomputed here."""
    img = [[[float((5 * i + 3 * j + c) % 7) for c in range(3)] for j in range(4)]
           for i in range(4)]
    W = [[float((a * 3 + b) % 5) - 2.0 for b in range(2)] for a in range(12)]
    result = geron_pvt(np.array(img), [{"patch_size": 2, "dim": 2, "W": np.array(W)}])
    assert isinstance(result, dict)
    assert result["output_shape"] == (2, 2, 2)
    tok = np.asarray(result["tokens"]).tolist()
    for gi in range(2):
        for gj in range(2):
            p = [img[2 * gi + u][2 * gj + v][c] for u in range(2) for v in range(2)
                 for c in range(3)]
            exp = [sum(p[a] * W[a][b] for a in range(12)) for b in range(2)]
            assert tok[gi][gj] == pytest.approx(exp, rel=0, abs=1e-12)


def test_hmpvt_edge():
    """Spatial reduction r: attention over N = 16 queries and N / r^2
    keys costs 16 * 16 / r^2; without it, 16 * 16."""
    for r in (1, 2, 4):
        res = geron_pvt(np.zeros((8, 8, 1)), [{"patch_size": 2, "dim": 4, "sr_ratio": r}])
        assert int(res["full_attention_cost"]) == 16 * 16
        assert int(res["attention_cost"]) == 16 * 16 // (r * r)


# --- appended: the module's own worked example as a gate -----------

import doctest as _doctest

import morie.fn.hmpvt as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
