"""Shared fixture for the IPTW/MSM family tests.

Hernan & Robins (2020) Ch. 20: a time-varying covariate that confounds
the second treatment and is itself caused by the first. The truth is
computed from explicit potential outcomes generated with the same noise
draws, so it is a number that was calculated rather than assumed.

Not a test module -- imported by test_lggvls, test_tdcvar, test_gentmt
and test_polkrn.
"""

import math

import pytest

from morie.fn import _array_core as np

N = 4000
RHO, GAM, TH0 = 1.0, 0.9, 0.6
TH1 = TH0 + GAM * RHO           # 1.5, so the cumulative MSM is correct


def expit(z):
    return 1.0 / (1.0 + math.exp(-z))


@pytest.fixture(scope="module")
def feedback():
    """Hernan & Robins Ch. 20: L1 confounds A1 and is caused by A0."""
    rng = np.random.default_rng(20260813)
    L0 = [rng.standard_normal() for _ in range(N)]
    eta = [rng.standard_normal() for _ in range(N)]
    eps = [0.7 * rng.standard_normal() for _ in range(N)]
    u0 = [float(rng.uniform()) for _ in range(N)]
    u1 = [float(rng.uniform()) for _ in range(N)]

    def L1_of(a0, i):
        return RHO * a0 + 0.6 * L0[i] + eta[i]

    def Y_of(a0, a1, i):
        return (1.0 + TH0 * a0 + TH1 * a1 + GAM * L1_of(a0, i)
                + 0.7 * L0[i] + eps[i])

    A0 = [1.0 if u0[i] < expit(-0.3 + 1.2 * L0[i]) else 0.0
          for i in range(N)]
    L1 = [L1_of(A0[i], i) for i in range(N)]
    A1 = [1.0 if u1[i] < expit(-0.2 + 1.0 * L1[i] + 0.8 * A0[i]) else 0.0
          for i in range(N)]
    Y = [Y_of(A0[i], A1[i], i) for i in range(N)]
    ey = {}
    for a0 in (0.0, 1.0):
        for a1 in (0.0, 1.0):
            ey[(a0, a1)] = sum(Y_of(a0, a1, i) for i in range(N)) / N
    return {"Y": Y, "A": [A0, A1],
            "L": [[[v] for v in L0], [[v] for v in L1]],
            "truth": (ey[(1.0, 1.0)] - ey[(0.0, 0.0)]) / 2.0,
            "EY": ey, "Y_of": Y_of, "rng": rng}


@pytest.fixture(scope="module")
def dose(feedback):
    """A continuous exposure in the finite-variance regime, slope 1.5."""
    rng = feedback["rng"]
    G1 = [rng.standard_normal() for _ in range(N)]
    G2 = [rng.standard_normal() for _ in range(N)]
    X = [[G1[i], G2[i]] for i in range(N)]
    D = [1.0 + 0.3 * G1[i] + 0.2 * G2[i] + rng.standard_normal()
         for i in range(N)]
    Y = [0.5 + 1.5 * D[i] + 2.0 * G1[i] - 1.0 * G2[i]
         + 0.5 * rng.standard_normal() for i in range(N)]
    return {"Y": Y, "D": D, "X": X}


