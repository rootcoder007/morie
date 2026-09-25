"""Tests for schN.schnet (SchNet continuous-filter convolution, Schutt et al. 2017)."""

import math

import pytest

from morie.fn.schN import cfconv, gaussian_expansion

X = [[1.0, 0.5], [0.2, -0.3], [-0.7, 1.1], [0.4, 0.4]]
R = [[0.0, 0.0, 0.0], [1.1, 0.2, -0.3], [0.3, 1.4, 0.5], [-0.8, 0.6, 2.2]]


def _net(e):
    # a fixed "filter network": two linear read-outs of the expansion
    return [sum(v * (k + 1) / 25 for k, v in enumerate(e)), sum(v * (-1) ** k for k, v in enumerate(e))]


def _fc(r, rc=5.0):
    return 0.5 * (math.cos(math.pi * r / rc) + 1.0) if r < rc else 0.0


def test_schN_basic():
    """x_i' = sum_{j != i} x_j * W(|r_i - r_j|) * f_c(|r_i - r_j|) with
    W = filter_net(gaussian expansion) and the cosine cutoff
    f_c = (cos(pi r / r_c) + 1) / 2, recomputed."""
    out = cfconv(X, R, _net, cutoff=5.0)
    for i in range(4):
        exp = [0.0, 0.0]
        for j in range(4):
            if i != j:
                r = math.dist(R[i], R[j])
                w = _net(gaussian_expansion(r))
                for a in range(2):
                    exp[a] += X[j][a] * w[a] * _fc(r)
        assert out[i] == pytest.approx(exp, rel=1e-13, abs=1e-15)


def test_schN_edge():
    """Rotating and translating every position leaves the output
    unchanged (positions enter only through distances); a filter of
    the wrong width is refused."""
    c, s = math.cos(0.9), math.sin(0.9)
    Rt = [[c * p[0] - s * p[1] + 3.0, s * p[0] + c * p[1] - 1.0, p[2] + 0.5] for p in R]
    a, b = cfconv(X, R, _net), cfconv(X, Rt, _net)
    for i in range(4):
        assert b[i] == pytest.approx(a[i], rel=1e-12, abs=1e-14)
    with pytest.raises(ValueError):
        cfconv(X, R, lambda e: [1.0])
    # the default basis: 25 centres on [0, 6], spacing 0.25, gamma = 1/(2 * 0.25^2)
    e = gaussian_expansion(1.3)
    assert e == pytest.approx([math.exp(-8.0 * (1.3 - 0.25 * k) ** 2) for k in range(25)], rel=1e-14)
