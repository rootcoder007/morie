"""Tests for morie.fn.bbdim — Blackbox dimensionality."""

from morie.fn import _array_core as np
from morie.fn.bbdim import bbdim


def test_bbdim_smoke():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((30, 5))
    r = bbdim(X, max_dims=4)
    assert r.name == "bb_dimensionality_select"
    assert 1 <= r.value <= 4
    assert "eigenvalues" in r.extra


def test_cheatsheet():
    from morie.fn.bbdim import cheatsheet

    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0


def test_bbdim_eigenvalues_and_elbow_recomputed():
    """Covariance eigenvalues of centred data (SVD route) and the largest
    drop between consecutive eigenvalues."""
    import math

    import pytest

    Z = [[1.0, 2.0], [2.0, 3.5], [3.0, 3.0], [4.0, 6.0], [5.0, 5.5]]
    n = 5
    m = [sum(r[j] for r in Z) / n for j in range(2)]
    s = [[sum((r[a] - m[a]) * (r[b] - m[b]) for r in Z) / (n - 1) for b in range(2)] for a in range(2)]
    tr, det = s[0][0] + s[1][1], s[0][0] * s[1][1] - s[0][1] ** 2
    disc = math.sqrt(tr * tr / 4 - det)
    ev = [tr / 2 + disc, tr / 2 - disc]
    r = bbdim(Z, max_dims=2)
    assert r.extra["eigenvalues"] == pytest.approx(ev, rel=1e-12)
    assert r.value == 1
