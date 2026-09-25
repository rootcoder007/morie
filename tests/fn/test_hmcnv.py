"""Tests for hmcnv.geron_convolutional_layer (Geron Eq. 12-1)."""

import pytest

from morie.fn.hmcnv import geron_convolutional_layer


def test_hmcnv_basic():
    """A 2x2 all-ones kernel over the 3x3 image 1..9 sums each 2x2 block:
    [[1+2+4+5, 2+3+5+6], [4+5+7+8, 5+6+8+9]] = [[12, 16], [24, 28]];
    a bias of 0.5 adds to every cell."""
    x = [[[float(3 * i + j + 1)] for j in range(3)] for i in range(3)]
    k = [[[[1.0]] for _ in range(2)] for _ in range(2)]
    result = geron_convolutional_layer(x, k, bias=[0.5])
    assert isinstance(result, dict)
    assert [[c[0] for c in row] for row in result["z"]] == [[12.5, 16.5], [24.5, 28.5]]
    assert (result["height"], result["width"], result["channels"]) == (2, 2, 1)
    assert result["nparams"] == 2 * 2 * 1 * 1 + 1


def test_hmcnv_edge():
    """Two in-channels, two out-channels, stride 2 and zero padding 1,
    against a direct evaluation of Eq. 12-1 on the zero-padded input."""
    h, w = 4, 5
    x = [[[float((i * 7 + j * 3 + c) % 5) - 2.0 for c in range(2)]
          for j in range(w)] for i in range(h)]
    k = [[[[float(u - v + c - o) * 0.5 for o in range(2)] for c in range(2)]
          for v in range(3)] for u in range(3)]
    b = [1.0, -1.0]
    r = geron_convolutional_layer(x, k, bias=b, stride=(2, 2), padding=(1, 1))
    xp = [[[0.0, 0.0] for _ in range(w + 2)] for _ in range(h + 2)]
    for i in range(h):
        for j in range(w):
            xp[i + 1][j + 1] = list(x[i][j])
    oh, ow = (h + 2 - 3) // 2 + 1, (w + 2 - 3) // 2 + 1
    assert (r["height"], r["width"]) == (oh, ow)
    for i in range(oh):
        for j in range(ow):
            for o in range(2):
                z = b[o] + sum(xp[2 * i + u][2 * j + v][c] * k[u][v][c][o]
                               for u in range(3) for v in range(3) for c in range(2))
                assert r["z"][i][j][o] == pytest.approx(z, rel=0, abs=1e-12)
    with pytest.raises(ValueError):
        geron_convolutional_layer(x, [[[[1.0]]]])   # in-channel mismatch
