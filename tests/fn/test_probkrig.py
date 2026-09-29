"""probkrig: unbiasedness constraints, exactness at data points and the kriging equations."""

import math

import pytest

from morie.fn._rng import random_uniform
from morie.fn.probkrig import probability_kriging

U = [float(v) for v in random_uniform(100, seed=5)]
P = [[10 * U[i], 10 * U[20 + i]] for i in range(20)]
Z = [math.cos(P[i][0] / 2) + P[i][1] / 4 + U[40 + i] for i in range(20)]
CI, CU, CIU = (0.0, 0.2, 2.5), (0.0, 0.08, 2.5), (0.0, 0.1, 2.5)


def test_constraints_and_exactness():
    thr = sorted(Z)[9]
    r = probability_kriging(P, Z, [[3.0, 4.0], P[7]], thr, CI, CU, CIU)
    w = r.weights[0]
    assert sum(w[:20]) == pytest.approx(1.0, abs=1e-10) and sum(w[20:]) == pytest.approx(0.0, abs=1e-10)
    # without nugget, cokriging interpolates the indicator at a data location
    assert r.estimate[1] == pytest.approx(r.indicator[7], abs=1e-9)
    assert sorted(r.uniform) == pytest.approx([(k + 0.5) / 20 for k in range(20)])
    assert all(0.0 <= p <= 1.0 for p in r.probability)
