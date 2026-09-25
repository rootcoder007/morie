"""Tests for rgica.rangayyan_fastica."""

import math

import pytest

from morie.fn.bsaclass import rangayyan_fastica


def _corr(a, b):
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    num = sum((u - ma) * (v - mb) for u, v in zip(a, b))
    return num / math.sqrt(sum((u - ma) ** 2 for u in a) * sum((v - mb) ** 2 for v in b))


def test_rgica_basic():
    """Two independent non-Gaussian sources (a square wave and a sawtooth)
    mixed by a known matrix are recovered up to sign and order."""
    n = 1000
    s1 = [1.0 if math.sin(0.05 * t) >= 0 else -1.0 for t in range(n)]
    s2 = [((t * 0.037) % 1.0) - 0.5 for t in range(n)]
    X = [[0.8 * a + 0.6 * b for a, b in zip(s1, s2)],
         [0.3 * a - 0.9 * b for a, b in zip(s1, s2)]]
    r = rangayyan_fastica(X, 2)
    src = [list(row) for row in r["sources"]]
    c = [[abs(_corr(src[i], s)) for s in (s1, s2)] for i in range(2)]
    assert max(c[0][0] * c[1][1], c[0][1] * c[1][0]) > 0.995


def test_rgica_edge():
    """Seeded runs repeat exactly; the unmixed sources are uncorrelated."""
    n = 600
    X = [[math.sin(0.11 * t) + 0.5 * (((t * 0.029) % 1.0) - 0.5) for t in range(n)],
         [0.4 * math.sin(0.11 * t) - (((t * 0.029) % 1.0) - 0.5) for t in range(n)]]
    a = rangayyan_fastica(X, 2, seed=3)
    b = rangayyan_fastica(X, 2, seed=3)
    assert [list(r) for r in a["sources"]] == [list(r) for r in b["sources"]]
    s = [list(r) for r in a["sources"]]
    assert abs(_corr(s[0], s[1])) < 1e-8


