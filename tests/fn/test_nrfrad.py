"""Tests for nrfrad.nerf_radiance (NeRF volume rendering, Mildenhall et
al. 2020, eq. 3)."""

import math

import pytest

from morie.fn.nrfrad import nerf_radiance

SIG = [0.0, 0.5, 2.0, 0.1, 3.0]
COL = [[1.0, 0.0, 0.0], [0.2, 0.4, 0.6], [0.0, 1.0, 0.0], [0.9, 0.9, 0.1], [0.0, 0.0, 1.0]]
T = [0.0, 0.3, 0.7, 1.2, 1.4]


def test_nrfrad_basic():
    """C = sum_i T_i alpha_i c_i with T_i = exp(-sum_{j<i} sigma_j d_j),
    alpha_i = 1 - exp(-sigma_i d_i), d_i = t_{i+1} - t_i and the last
    interval 1e10 (the paper's code), recomputed."""
    r = nerf_radiance(SIG, COL, T)
    assert isinstance(r, dict)
    d = [T[i + 1] - T[i] for i in range(4)] + [1e10]
    w, acc = [], 0.0
    for i in range(5):
        w.append(math.exp(-acc) * (1 - math.exp(-SIG[i] * d[i])))
        acc += SIG[i] * d[i]
    assert r["weights"] == pytest.approx(w, rel=1e-14, abs=1e-16)
    assert r["colour"] == pytest.approx(
        [sum(w[i] * COL[i][c] for i in range(5)) for c in range(3)], rel=1e-14)
    # the infinite last interval makes the ray fully opaque
    assert r["accumulated_alpha"] == pytest.approx(1.0, rel=1e-14)


def test_nrfrad_edge():
    """Empty space renders nothing until a density is met; negative
    density is refused."""
    r = nerf_radiance([0.0, 0.0], [[1.0, 1.0, 1.0], [0.5, 0.5, 0.5]], [0.0, 1.0])
    assert r["colour"] == [0.0, 0.0, 0.0]
    assert r["transmittance_final"] == 1.0
    with pytest.raises(ValueError):
        nerf_radiance([-1.0], [[0.0, 0.0, 0.0]], [0.0])
