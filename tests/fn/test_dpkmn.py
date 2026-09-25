"""Tests for dpkmn.dp_kmeans (Laplace-mechanism Lloyd iterations)."""

import math

import pytest

from morie.fn.dpkmn import dp_kmeans


def _two_blobs(n=100):
    pts = []
    for k in range(n):
        c = -2.0 if k % 2 else 2.0
        pts.append([c + 0.3 * math.sin(1.7 * k), c + 0.3 * math.cos(2.3 * k)])
    return pts


def test_dpkmn_basic():
    """With an enormous budget the noise vanishes and the released
    centres are Lloyd's: the two blob means.  The budget is split over
    iterations and between the sum and count queries."""
    X = _two_blobs()
    r = dp_kmeans(X, k=2, epsilon=1e9, n_iter=6, bounds=(-5, 5), seed=4)
    c = sorted([list(map(float, row)) for row in r["centers"]])
    m_neg = [sum(p[j] for p in X if p[0] < 0) / 50 for j in range(2)]
    m_pos = [sum(p[j] for p in X if p[0] > 0) / 50 for j in range(2)]
    assert c[0] == pytest.approx(m_neg, abs=1e-6)
    assert c[1] == pytest.approx(m_pos, abs=1e-6)
    assert r["epsilon_per_iteration"] == pytest.approx(1e9 / 12, rel=1e-15)
    assert r["private_outputs"] == ("centers",)


def test_dpkmn_edge():
    """The same seed reproduces the release; k < 1 and n_iter < 1 raise;
    omitting bounds is flagged as a non-private query."""
    X = _two_blobs()
    a = dp_kmeans(X, k=2, epsilon=2.0, bounds=(-5, 5), seed=7)
    b = dp_kmeans(X, k=2, epsilon=2.0, bounds=(-5, 5), seed=7)
    assert [list(map(float, r)) for r in a["centers"]] == [list(map(float, r)) for r in b["centers"]]
    with pytest.raises(ValueError):
        dp_kmeans(X, k=0)
    with pytest.raises(ValueError):
        dp_kmeans(X, n_iter=0)
    r = dp_kmeans(X, k=2, epsilon=2.0, seed=1)
    assert any("non-private" in w for w in r.warnings)
