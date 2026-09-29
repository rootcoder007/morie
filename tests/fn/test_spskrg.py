"""Tests for spskrg.schabenberger_simple_kriging: the kriging equations recomputed."""

import math

from morie.fn import _array_core as np
from morie.fn.spskrg import schabenberger_simple_kriging

P = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.4, 1.7]]
Z = [1.0, 2.0, 1.5, 3.0, 2.2]
T = [[0.5, 0.5]]


def _cov(h):
    # exponential model with practical range 1: exp(-3 h / range) (Schabenberger and Gotway eq. 4.11)
    return math.exp(-3.0 * h)


def test_predictor_and_variance():
    S = np.array([[_cov(math.dist(p, q)) for q in P] for p in P])
    s = np.array([_cov(math.dist(p, T[0])) for p in P])
    lam = np.linalg.inv(S) @ s
    r = schabenberger_simple_kriging(P, Z, T, mu=1.8)
    assert abs(float(r["prediction"][0]) - (1.8 + float(lam @ (np.array(Z) - 1.8)))) < 1e-12
    assert abs(float(r["variance"][0]) - (1.0 - float(s @ lam))) < 1e-12


def test_exact_interpolation():
    r = schabenberger_simple_kriging(P, Z, [P[3]], mu=1.8)
    assert abs(float(r["prediction"][0]) - 3.0) < 1e-12
    assert abs(float(r["variance"][0])) < 1e-12
