"""Tests for morie.fn.carsim: draws solve L' x = z for the Philox normals."""

from morie.fn import _array_core as np
from morie.fn._rng import random_normal
from morie.fn.carsim import carsim

W = [[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]]


def test_precision_whitening():
    rho, s2 = 0.3, 2.0
    r = carsim(W, rho, s2, nsim=3, seed=7)
    Q = (np.eye(4) - rho * np.array(W)) / s2
    for k, x in enumerate(r["draws"]):
        z = [float(v) for v in random_normal(4, seed=7, stream=k)]
        # x' Q x = z'z because x = L'^{-1} z and Q = L L'
        xa = np.array(x)
        assert abs(float(xa @ (Q @ xa)) - sum(v * v for v in z)) < 1e-12
