"""stcokrig: cokriging constraints, exactness and the diffusion covariance identities."""

import math

import pytest

from morie.fn._rng import random_uniform
from morie.fn.stcokrig import diffusion_st_covariance, st_cokriging

U = [float(v) for v in random_uniform(120, seed=9)]
P = [[10 * U[i], 10 * U[20 + i]] for i in range(20)]
T = [float(int(3 * U[40 + i])) for i in range(20)]
V = [i % 2 for i in range(20)]
Z = [math.cos(P[i][0] / 3) + 0.1 * T[i] + U[60 + i] for i in range(20)]
B = [[1.0, 0.5], [0.5, 0.9]]


def test_cokriging_exactness_and_variance():
    i0 = V.index(0)
    r = st_cokriging(P, T, V, Z, [P[i0]], [T[i0]], B, 3.0, 2.0)
    assert r.estimate[0] == pytest.approx(Z[i0], abs=1e-9)
    assert r.variance[0] == pytest.approx(0.0, abs=1e-9)
    far = st_cokriging(P, T, V, Z, [[1e4, 1e4]], [100.0], B, 3.0, 2.0)
    m0 = [Z[i] for i in range(20) if V[i] == 0]
    # far from all data, ordinary cokriging returns the generalized least squares mean of variable 0
    assert far.variance[0] > 1.0
    assert min(m0) - 1 < far.estimate[0] < max(m0) + 1


def test_diffusion_covariance():
    assert diffusion_st_covariance(0.0, 0.0, 2.0, 1.5, 0.7) == pytest.approx(2.0, rel=1e-15)
    h, u, xi, D = 1.3, 0.8, 1.1, 0.4
    s = xi * xi + 4 * D * u
    assert diffusion_st_covariance(h, u, 1.0, xi, D, 2) == pytest.approx(xi * xi / s * math.exp(-h * h / s), rel=1e-14)
    assert diffusion_st_covariance([h], [-u], 1.0, xi, D, 2)[0] == pytest.approx(
        diffusion_st_covariance(h, u, 1.0, xi, D, 2)
    )
