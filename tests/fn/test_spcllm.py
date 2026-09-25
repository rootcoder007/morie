"""Tests for spcllm: local Moran's I with Anselin's quadrant labels.

The statistic is Schabenberger and Gotway (2005) eq. (1.17):
I(s_i) = n / ((n-1) S^2) (Z_i - Zbar) sum_j w_ij (Z_j - Zbar).
Its conditional-randomization moments are checked against a brute-force
permutation of the other sites.
"""

import itertools
import math
import random

import pytest

from morie.fn.spcllm import spatial_cluster_lisa


def _rook(r, c):
    n = r * c
    W = [[0.0] * n for _ in range(n)]
    for i in range(r):
        for j in range(c):
            for di, dj in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                a, b = i + di, j + dj
                if 0 <= a < r and 0 <= b < c:
                    W[i * c + j][a * c + b] = 1.0
    return W


W9 = _rook(3, 3)
X9 = [5.0, 4.0, 1.0, 4.5, 3.0, 0.5, 2.0, 1.0, 0.0]


def _local(x, W):
    n = len(x); m = sum(x) / n
    S2 = sum((v - m) ** 2 for v in x) / (n - 1)
    return [n / ((n - 1) * S2) * (x[i] - m) * sum(W[i][j] * (x[j] - m) for j in range(n))
            for i in range(n)]


def test_spcllm_basic():
    """The local statistic is eq. (1.17)."""
    r = spatial_cluster_lisa(X9, W9, 0.05)
    assert list(r["local"]) == pytest.approx(_local(X9, W9), rel=1e-12)
    assert r["n"] == 9


def test_conditional_randomization_moments_match_brute_force():
    """Hold site i fixed, permute the rest: the exact mean and variance."""
    r = spatial_cluster_lisa(X9, W9, 0.05)
    n = 9; m = sum(X9) / n
    S2 = sum((v - m) ** 2 for v in X9) / (n - 1)
    for i in (0, 4, 8):
        others = [X9[k] for k in range(n) if k != i]
        nbr = [k for k in range(n) if k != i]
        vals = []
        for perm in itertools.permutations(others):
            xs = list(X9)
            for k, v in zip(nbr, perm):
                xs[k] = v
            vals.append(n / ((n - 1) * S2) * (X9[i] - m)
                        * sum(W9[i][j] * (xs[j] - m) for j in range(n)))
        mu = sum(vals) / len(vals)
        var = sum((v - mu) ** 2 for v in vals) / len(vals)
        # the book's mean: -(n-1)^-1 sum_j w_ij (up to the fixed factor)
        want_z = (r["local"][i] - mu) / math.sqrt(var)
        assert r["z"][i] == pytest.approx(want_z, rel=1e-9)


def test_labels_follow_the_moran_scatterplot_and_significance():
    r = spatial_cluster_lisa(X9, W9, 0.05)
    m = sum(X9) / 9
    for i, lab in enumerate(r["labels"]):
        if r["p_value"][i] >= 0.05:
            assert lab == "NS"
        else:
            hi = X9[i] > m
            lag_hi = r["lagged_mean"][i] > 0
            assert lab == ("H" if hi else "L") + ("H" if lag_hi else "L")
    assert sum(r["counts"].values()) == 9
    assert r["counts"]["HH"] == sum(1 for l in r["labels"] if l == "HH")


def test_spcllm_edge():
    with pytest.raises(ValueError, match="square"):
        spatial_cluster_lisa(X9, [[0.0, 1.0]], 0.05)
