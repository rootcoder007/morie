"""Tests for splisa.schabenberger_lisa.

Reference values are spdep 1.3 localmoran() and moran.test() on the same
grid with binary (style "B") weights.
"""

import math

import pytest

NR, NC = 5, 4
N = NR * NC
# rook adjacency on a 5 x 4 grid, sites numbered row by row
W = [[1.0 if abs(i // NC - j // NC) + abs(i % NC - j % NC) == 1 else 0.0
      for j in range(N)] for i in range(N)]
Z = [((i * 37) % 17) / 3 + 0.5 * (i // NC) for i in range(N)]
X1 = [float((i * 5) % 9) for i in range(N)]
X2 = [math.cos(i) for i in range(N)]
Y = [1 + 0.8 * a - 0.5 * b + ((i * 13) % 7 - 3) / 2 + 0.3 * (i // NC)
     for i, (a, b) in enumerate(zip(X1, X2))]
X = [[1.0, a, b] for a, b in zip(X1, X2)]

from morie.fn.splisa import schabenberger_lisa


def test_splisa_basic():
    """Local and global Moran match spdep; E_r is -w_i./(n-1)."""
    r = schabenberger_lisa(Z, W)
    spdep_ii = [1.3857203210815374, 2.0215462610899868,
                2.3341782847486265, 0.38022813688212914]
    for got, ref in zip(r["local"][:4], spdep_ii):
        assert got == pytest.approx(ref, rel=1e-12)
    assert r["global_i"] == pytest.approx(0.087697779958297514, rel=1e-12)
    for i in range(N):
        assert r["expectation"][i] == pytest.approx(-sum(W[i]) / (N - 1), rel=1e-15)
    # sum_i I(s_i) = w.. I, eq (1.17)
    assert abs(r["sum_identity_gap"]) < 1e-12


def test_splisa_edge():
    """Fewer than three sites is rejected."""
    with pytest.raises(ValueError, match="at least 3 sites"):
        schabenberger_lisa([1.0, 2.0], [[0.0, 1.0], [1.0, 0.0]])
