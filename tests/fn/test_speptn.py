"""Tests for speptn.spatial_pca.

Reference eigenvalues are numpy.linalg.eigh of H = Z'((W+W')/2)Z/n.
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

from morie.fn.speptn import spatial_pca

DATA = [[Z[i], X1[i], X2[i], Y[i]] for i in range(N)]


def test_speptn_basic():
    """All four axes match numpy, ordered most positive first as in ade4."""
    r = spatial_pca(DATA, W, 4)
    ref = [0.29308673815833364, 0.013965248377333123,
           -0.35457620488082386, -1.0826592050025354]
    for got, want in zip(r["eigenvalues"], ref):
        assert got == pytest.approx(want, abs=1e-12)
    assert r["total_variance"] == 4.0


def test_speptn_edge():
    """naxes=1 keeps the most positive axis; scores and lags follow from it."""
    r = spatial_pca(DATA, W, 1)
    assert r["eigenvalues"][0] == pytest.approx(0.29308673815833364, abs=1e-12)
    v = r["loadings"][0]
    assert sum(t * t for t in v) == pytest.approx(1.0, abs=1e-12)
    cols = list(zip(*DATA))
    zc = []
    for c in cols:
        m = sum(c) / N
        s = math.sqrt(sum((t - m) ** 2 for t in c) / N)
        zc.append([(t - m) / s for t in c])
    sc = [sum(zc[j][i] * v[j] for j in range(4)) for i in range(N)]
    for got, want in zip(r["scores"][0], sc):
        assert got == pytest.approx(want, abs=1e-12)
    lag = [sum(W[i][j] * sc[j] for j in range(N)) for i in range(N)]
    for got, want in zip(r["lagged_scores"][0], lag):
        assert got == pytest.approx(want, abs=1e-12)
