"""Tests for jotide.joseph_tide_encoder."""

import math

import pytest

from morie.fn.jotide import joseph_tide_encoder


def _w(r, c, s):
    return [[math.sin(s + 0.7 * i + 1.3 * j) * 0.5 for j in range(c)] for i in range(r)]


def _ln(v):
    m = sum(v) / len(v)
    return [x - m for x in v]


def _zero_block(n_in, n_hid, n_out):
    return ([[0.0] * n_in] * n_hid, [0.0] * n_hid, [[0.0] * n_hid] * n_out, [0.0] * n_out,
            [[0.0] * n_in] * n_out)


def test_jotide_basic():
    """With every residual block at zero weight the dense path outputs
    LayerNorm(0) = 0, so the forecast is exactly the global linear map of
    the lookback that TiDE adds to its output."""
    y = [1.0, 3.0, 2.0, 5.0]
    wg = [[0.1, 0.2, 0.3, 0.4], [0.5, -0.5, 0.0, 1.0]]
    r = joseph_tide_encoder(y, [], None, _zero_block(4, 3, 4), _zero_block(4, 3, 6),
                            _zero_block(3, 2, 1), wg, 2)
    assert r["forecast"] == pytest.approx([sum(a * b for a, b in zip(w, y)) for w in wg], rel=1e-14)
    assert (r["p"], r["horizon"]) == (3, 2)


def test_jotide_edge():
    y = [1.0, 3.0, 2.0, 5.0]
    with pytest.raises(ValueError, match="multiple of horizon"):
        joseph_tide_encoder(y, [], None, _zero_block(4, 3, 4), _zero_block(4, 3, 5),
                            _zero_block(3, 2, 1), [[0.0] * 4] * 2, 2)


