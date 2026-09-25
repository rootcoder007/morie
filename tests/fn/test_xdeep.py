"""Tests for xdeep.xdeepfm (Compressed Interaction Network, Lian et al. 2018)."""

import math

import pytest

from morie.fn.xdeep import cin_layer, interaction_degree, xdeepfm


X0 = [[math.sin(0.7 * i + 0.3 * a) for a in range(4)] for i in range(3)]   # m = 3 fields, D = 4


def _W(H, mp, m, s):
    return [[[math.cos(s + 1.1 * h + 0.7 * i + 0.3 * j) for j in range(m)] for i in range(mp)]
            for h in range(H)]


def test_xdeep_basic():
    """X^k_{h,a} = sum_ij W^k_{hij} X^{k-1}_{i,a} X^0_{j,a}; each layer is
    sum-pooled over the embedding axis.  A second layer is a cubic form
    in X^0, which the degree bookkeeping reports as 3."""
    W1, W2 = _W(2, 3, 3, 0.0), _W(2, 2, 3, 1.0)
    r = xdeepfm(X0, [W1, W2])
    L1 = [[sum(W1[h][i][j] * X0[i][a] * X0[j][a] for i in range(3) for j in range(3)) for a in range(4)]
          for h in range(2)]
    L2 = [[sum(W2[h][i][j] * L1[i][a] * X0[j][a] for i in range(2) for j in range(3)) for a in range(4)]
          for h in range(2)]
    assert r["layers"][0] == [pytest.approx(row, abs=1e-14) for row in L1]
    assert r["layers"][1] == [pytest.approx(row, abs=1e-14) for row in L2]
    assert r["pooled"] == pytest.approx([sum(v) for v in L1] + [sum(v) for v in L2], abs=1e-13)
    assert r["degrees"] == [2, 3]
    # scaling X^0 by c scales layer k by c^(k+1): degree k+1 exactly
    r2 = xdeepfm([[2 * v for v in row] for row in X0], [W1, W2])
    assert r2["pooled"][2:] == pytest.approx([8 * v for v in r["pooled"][2:]], rel=1e-12)


def test_xdeep_edge():
    """Mismatched embedding sizes raise; negative layer indices raise."""
    with pytest.raises(ValueError):
        cin_layer([[1.0, 2.0]], [[1.0, 2.0, 3.0]], [[[1.0]]])
    with pytest.raises(ValueError):
        interaction_degree(-1)
