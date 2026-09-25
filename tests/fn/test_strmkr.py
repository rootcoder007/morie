"""Tests for strmkr.strauss_process (Strauss process by pseudolikelihood)."""

import math

import pytest

from morie.fn.strmkr import strauss_process


def _pattern(n=40):
    """A regular-ish pattern: jittered lattice with a deterministic hash."""
    P = []
    for k in range(n):
        u = ((math.sin(12.9898 * k + 4.1) * 43758.5453) % 1)
        v = ((math.sin(78.233 * k + 1.7) * 12345.678) % 1)
        P.append([((k % 7) + 0.3 + 0.4 * u) / 7.0, ((k // 7) + 0.3 + 0.4 * v) / 6.0])
    return P


def _scheme(P, r, nx, ny, win=(0.0, 1.0, 0.0, 1.0)):
    """Berman-Turner quadrature: data plus the centres of an nx x ny
    grid, counting weights (tile area / points in the tile), and
    t(u) = number of data points within r of u, excluding u itself."""
    n = len(P)
    D = [[win[0] + (a + 0.5) * (win[1] - win[0]) / nx, win[2] + (b + 0.5) * (win[3] - win[2]) / ny]
         for a in range(nx) for b in range(ny)]
    Q = P + D
    tile = [(min(int((q[0] - win[0]) / (win[1] - win[0]) * nx), nx - 1),
             min(int((q[1] - win[2]) / (win[3] - win[2]) * ny), ny - 1)) for q in Q]
    cnt = {t: tile.count(t) for t in set(tile)}
    area = (win[1] - win[0]) * (win[3] - win[2])
    w = [area / (nx * ny) / cnt[t] for t in tile]
    t = [sum(1 for j in range(n) if j != i and math.dist(q, P[j]) < r) for i, q in enumerate(Q)]
    z = [1.0] * n + [0.0] * len(D)
    return w, t, z


def test_strmkr_basic():
    """The fit solves the weighted Poisson score equations
    sum_i w_i (z_i/w_i - exp(b0 + b1 t_i)) (1, t_i) = 0 of the rebuilt
    scheme, and the standard errors are the inverse of
    sum_i w_i mu_i (1, t_i)(1, t_i)'.  The sufficient statistic is the
    count of pairs closer than r.  On this pattern spatstat's
    ppm(Q ~ 1, Strauss(0.12), correction = "none") with the same 10 x 10
    dummy grid gives log beta 5.160381, log gamma -2.290515 and
    vcov(hessian = TRUE) standard errors 0.1682247, 0.3829443."""
    P = _pattern()
    r0 = 0.12
    out = strauss_process(P, r0, window=(0, 1, 0, 1), nx=10, ny=10)
    w, t, z = _scheme(P, r0, 10, 10)
    b0, b1 = out["log_beta"], out["log_gamma"]
    mu = [math.exp(b0 + b1 * ti) for ti in t]
    s0 = sum(zi - wi * m for zi, wi, m in zip(z, w, mu))
    s1 = sum((zi - wi * m) * ti for zi, wi, m, ti in zip(z, w, mu, t))
    assert abs(s0) < 1e-8 and abs(s1) < 1e-8
    I00 = sum(wi * m for wi, m in zip(w, mu))
    I01 = sum(wi * m * ti for wi, m, ti in zip(w, mu, t))
    I11 = sum(wi * m * ti * ti for wi, m, ti in zip(w, mu, t))
    det = I00 * I11 - I01 * I01
    assert out["se_log_beta"] == pytest.approx(math.sqrt(I11 / det), rel=1e-9)
    assert out["se_log_gamma"] == pytest.approx(math.sqrt(I00 / det), rel=1e-9)
    pairs = sum(1 for i in range(len(P)) for j in range(i + 1, len(P)) if math.dist(P[i], P[j]) < r0)
    assert out["n_close_pairs"] == pairs
    assert out["gamma"] == pytest.approx(math.exp(b1), rel=1e-15)
    assert out["gamma"] < 1.0 and out["valid_density"] is True
    assert out["n_quadrature"] == len(P) + 100


def test_strmkr_edge():
    """An empty pattern, a non-positive radius and a degenerate window
    raise."""
    P = _pattern()
    with pytest.raises(ValueError):
        strauss_process([], 0.1)
    with pytest.raises(ValueError):
        strauss_process(P, 0.0)
    with pytest.raises(ValueError):
        strauss_process(P, 0.1, window=(0, 0, 0, 1))
