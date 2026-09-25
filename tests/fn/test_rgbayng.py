"""Tests for rgbayng.rangayyan_bayes_gaussian."""

import math

import pytest

from morie.fn.bsaclass import rangayyan_bayes_gaussian


def _logpdf2(x, m, C):
    det = C[0][0] * C[1][1] - C[0][1] * C[1][0]
    inv = [[C[1][1] / det, -C[0][1] / det], [-C[1][0] / det, C[0][0] / det]]
    d = [x[0] - m[0], x[1] - m[1]]
    q = sum(d[a] * inv[a][b] * d[b] for a in range(2) for b in range(2))
    return -math.log(2 * math.pi) - 0.5 * math.log(det) - 0.5 * q


def test_rgbayng_basic():
    """Eq. (10.72): d_i(x) = ln P(C_i) + ln N(x; m_i, C_i), the full normal
    log-density recomputed here; x goes to the larger discriminant."""
    means = [[0.0, 0.0], [2.0, 1.0]]
    covs = [[[1.0, 0.3], [0.3, 2.0]], [[0.5, 0.0], [0.0, 0.5]]]
    x = [1.2, 0.4]
    r = rangayyan_bayes_gaussian(x, means, covs, priors=[0.7, 0.3])
    want = [math.log(p) + _logpdf2(x, m, C) for p, m, C in zip((0.7, 0.3), means, covs)]
    assert r["d_full"] == pytest.approx(want, rel=1e-12)
    assert r["assigned"] == want.index(max(want))


def test_rgbayng_edge():
    """Dropping (n/2) ln 2 pi (eq. 10.73) shifts every class equally, so
    the decision is the same."""
    means = [[0.0, 0.0], [2.0, 1.0]]
    covs = [[[1.0, 0.3], [0.3, 2.0]], [[0.5, 0.0], [0.0, 0.5]]]
    r = rangayyan_bayes_gaussian([1.5, 1.0], means, covs)
    shift = [a - b for a, b in zip(r["d_full"], r["d_dropped_constant"])]
    assert shift == pytest.approx([-math.log(2 * math.pi)] * 2, rel=1e-12)


