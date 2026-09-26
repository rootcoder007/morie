"""Tests for sirtdy.sir_age_structured (age-stratified SIR, next-generation R0)."""

import math

import pytest

from morie.fn.sirtdy import sir_age_structured


def test_sirtdy_basic():
    """One group: R0 = C S0 / (gamma N), and since dS/dR = -C S / (gamma N)
    the final state satisfies ln(S_inf / S0) = -(C / (gamma N)) (R_inf - R(0))
    exactly; RK4 with dt = 0.05 reproduces it to integration accuracy.
    Two groups: R0 is the spectral radius of K_ij = S_i C_ij / (gamma N_j),
    here the closed-form larger eigenvalue of a 2 x 2 matrix."""
    N, I0, C, g = 1000.0, 1.0, 0.3, 0.1
    r = sir_age_structured([N - I0], [I0], [0.0], [[C]], g, t_max=600.0, dt=0.05)
    assert r["R0"] == pytest.approx(C * (N - I0) / (g * N), rel=1e-12)
    s_inf, r_inf = r["S"][0], r["R"][0]
    # RK4 local error O(dt^5) accumulated over 12,000 steps: ~1e-9 relative
    assert math.log(s_inf / (N - I0)) == pytest.approx(-(C / (g * N)) * r_inf, rel=1e-7)
    assert r["estimate"] == pytest.approx(r_inf / N, rel=1e-15)
    S, Cm = [600.0, 400.0], [[0.4, 0.1], [0.2, 0.3]]
    q = sir_age_structured(S, [0.0, 0.0], [0.0, 0.0], Cm, 0.25, t_max=1.0)
    K = [[S[a] * Cm[a][b] / (0.25 * S[b]) for b in range(2)] for a in range(2)]
    tr, det = K[0][0] + K[1][1], K[0][0] * K[1][1] - K[0][1] * K[1][0]
    assert q["R0"] == pytest.approx(tr / 2 + math.sqrt(tr * tr / 4 - det), rel=1e-12)


def test_sirtdy_edge():
    """A non-square contact matrix and an empty group raise."""
    with pytest.raises(ValueError):
        sir_age_structured([10.0, 10.0], [1.0, 1.0], [0.0, 0.0], [[0.1, 0.2]], 0.1)
    with pytest.raises(ValueError):
        sir_age_structured([0.0], [0.0], [0.0], [[0.1]], 0.1)
