"""Tests for eslsoc.esl_self_organize."""

import math

import pytest

from morie.fn.eslsoc import esl_self_organize


def _clusters():
    pts = []
    for cx, cy in ((0.0, 0.0), (5.0, 5.0), (0.0, 5.0)):
        for k in range(40):
            pts.append([cx + 0.3 * math.cos(1.7 * k), cy + 0.3 * math.sin(2.3 * k)])
    return pts


def test_eslsoc_basic():
    """Every point is assigned to its nearest prototype, and the counts add
    up; the map quantises far better than the global mean."""
    X = _clusters()
    r = esl_self_organize(X, grid=(3, 3), seed=2)
    P = [list(p) for p in r["prototypes"].tolist()]
    asg = [int(a) for a in (r["assignment"].tolist() if hasattr(r["assignment"], "tolist") else r["assignment"])]
    for x, a in zip(X, asg):
        d = [sum((u - v) ** 2 for u, v in zip(x, p)) for p in P]
        assert d[a] == pytest.approx(min(d), rel=1e-12, abs=1e-15)
    assert sum(int(c) for c in (r["counts"].tolist() if hasattr(r["counts"], "tolist") else r["counts"])) == len(X)
    m = [sum(x[j] for x in X) / len(X) for j in range(2)]
    qe_mean = sum(math.sqrt(sum((x[j] - m[j]) ** 2 for j in range(2))) for x in X) / len(X)
    assert r["quantization_error"] < 0.2 * qe_mean


def test_eslsoc_edge():
    """Lattice neighbours stay close in data space: low topographic error."""
    r = esl_self_organize(_clusters(), grid=(3, 3), seed=2)
    assert r["topographic_error"] < 0.2


