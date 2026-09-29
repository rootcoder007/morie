"""Tests for morie.fn.bsthy — basic space dimensionality test."""

from morie.fn import _array_core as np
from morie.fn.bsthy import bsthy


def test_bsthy_smoke():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 4))
    r = bsthy(X, max_dims=3)
    assert r.name == "basic_space_dim_test"
    assert "eigenvalues" in r.extra
    assert len(r.extra["variance_ratios"]) == 3


def test_cheatsheet():
    from morie.fn.bsthy import cheatsheet

    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0


def test_bsthy_eigenvalue_ratios_recomputed():
    import math

    import pytest

    X = [[1.0, 2.0], [2.0, 3.5], [3.0, 3.0], [4.0, 6.0], [5.0, 5.5], [2.5, 1.0]]
    n = 6
    m = [sum(r[j] for r in X) / n for j in range(2)]
    s = [[sum((r[a] - m[a]) * (r[b] - m[b]) for r in X) / (n - 1) for b in range(2)] for a in range(2)]
    tr, det = s[0][0] + s[1][1], s[0][0] * s[1][1] - s[0][1] ** 2
    d = math.sqrt(tr * tr / 4 - det)
    ev = [tr / 2 + d, tr / 2 - d]
    r = bsthy(X, max_dims=2)
    assert r.extra["eigenvalues"] == pytest.approx(ev, rel=1e-12)
    assert r.extra["variance_ratios"] == pytest.approx([e / sum(ev) for e in ev], rel=1e-12)
    ratios = [e / sum(ev) for e in ev] + [0.0]
    drops = [ratios[i + 1] - ratios[i] < -0.1 for i in range(2)]
    assert r.value == drops.index(True) + 1
