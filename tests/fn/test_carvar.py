"""Tests for morie.fn.carvar: Sigma = sigma2 (I - rho W)^{-1} M."""

from morie.fn import _array_core as np
from morie.fn.carvar import carvar

W = [[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]]


def test_identity_form():
    S = np.array(carvar(W, 0.3, 1.7)["covariance"])
    ref = 1.7 * np.linalg.inv(np.eye(4) - 0.3 * np.array(W))
    assert float(abs(S - ref).max()) < 1e-14


def test_weighted_form_is_symmetric():
    d = [2.0, 2.0, 2.0, 2.0]
    Wr = [[v / d[i] for v in r] for i, r in enumerate(W)]
    r = carvar(Wr, 0.6, 1.0, m=[1 / v for v in d])
    ref = np.linalg.inv(np.diag(np.array(d)) - 0.6 * np.array(W))
    assert float(abs(np.array(r["covariance"]) - ref).max()) < 1e-14
