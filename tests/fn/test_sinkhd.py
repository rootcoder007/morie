"""Tests for sinkhd.sinkhorn_distance."""

import pytest

from morie.fn.sinkhd import sinkhorn_distance


A = [0.2, 0.3, 0.1, 0.4]
B = [0.25, 0.25, 0.5]
XA = [0.0, 1.0, 2.5, 4.0]
XB = [0.5, 2.0, 3.5]
C = [[(u - v) ** 2 for v in XB] for u in XA]


def test_sinkhd_basic():
    """The transport cost <P, C> and entropy of the converged Sinkhorn plan
    equal POT 0.9 ot.sinkhorn(a, b, C, reg=eps) at eps = 1."""
    r = sinkhorn_distance(A, B, C, 1.0, max_iter=5000)
    assert r["estimate"] == pytest.approx(0.6562052662258965, rel=1e-12)
    assert r["entropy"] == pytest.approx(2.6234129382290825, rel=1e-12)
    assert r["objective"] == pytest.approx(r["estimate"] - 1.0 * r["entropy"], rel=1e-15)
    assert r["lambda_"] == 1.0


def test_sinkhd_edge():
    """Stronger regularisation spreads the plan (more entropy, more cost);
    eps must be positive."""
    lo = sinkhorn_distance(A, B, C, 0.5, max_iter=5000)
    hi = sinkhorn_distance(A, B, C, 3.0, max_iter=5000)
    assert lo["estimate"] == pytest.approx(0.5271920848665448, rel=1e-12)
    assert hi["estimate"] == pytest.approx(1.2845140653037752, rel=1e-12)
    assert lo["entropy"] < hi["entropy"]
    with pytest.raises(ValueError, match="positive"):
        sinkhorn_distance(A, B, C, 0.0)


