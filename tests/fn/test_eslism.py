"""Tests for eslism.esl_isomap."""

import math

import pytest

from morie.fn.eslism import esl_isomap


def _chain(n):
    """Points evenly spaced along a straight line in the plane."""
    return [[float(i), 0.0] for i in range(n)]


def test_eslism_basic():
    """A one-dimensional chain embeds onto its own arclength."""
    n = 20
    X = _chain(n)
    result = esl_isomap(X, k=1, neighbors=2)

    emb = result["embedding"]
    assert emb.shape == (n, 1)
    assert result["neighbors"] == 2
    assert result["n_components"] == 1

    # Along a chain the graph distance is the sum of the unit steps, which for
    # this configuration is the straight-line distance itself.
    G = result["geodesic"]
    for i in range(n):
        for j in range(n):
            assert abs(G[i][j] - abs(i - j)) < 1e-9

    # The embedding is a one-dimensional reparameterisation of the index, so
    # it correlates perfectly (up to sign) with it.
    xs = [float(i) for i in range(n)]
    ys = [float(emb[i][0]) for i in range(n)]
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    sy = math.sqrt(sum((b - my) ** 2 for b in ys))
    assert abs(abs(cov / (sx * sy)) - 1.0) < 1e-6

    # Classical MDS on exact one-dimensional distances leaves no residual.
    assert result["residual_variance"] < 1e-9


def test_eslism_edge():
    """Geodesics dominate chords; bad neighbourhood sizes are refused."""
    # An L shape: the graph path from one arm to the other must go round the
    # corner, so it is strictly longer than the chord.
    X = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [2.0, 2.0]]
    result = esl_isomap(X, k=2, neighbors=1)
    G = result["geodesic"]
    # (0, 0) to (2, 2): along the graph 2 + 2 = 4, chord 2*sqrt(2).
    assert abs(G[0][4] - 4.0) < 1e-9
    assert G[0][4] > math.sqrt(8.0)

    # Every geodesic is at least the straight-line distance.
    for i in range(5):
        for j in range(5):
            chord = math.dist(X[i], X[j])
            assert G[i][j] >= chord - 1e-9

    # Two far-apart pairs with neighbors=1 leave the graph disconnected.
    with pytest.raises(ValueError, match="disconnected components"):
        esl_isomap([[0.0, 0.0], [0.0, 1.0], [50.0, 50.0], [50.0, 51.0]],
                   k=1, neighbors=1)

    with pytest.raises(ValueError, match="k must be between"):
        esl_isomap(X, k=0, neighbors=2)
    with pytest.raises(ValueError, match="neighbors must be between"):
        esl_isomap(X, k=1, neighbors=5)
