"""Tests for morie.fn.xrspz: spatial_zip redirects to the real estimator."""

from morie.fn.spcount import sar_zip
from morie.fn.xrspz import spat, spatial_zip, spatialzip

W = [[1.0 if abs(i - j) == 1 else 0.0 for j in range(10)] for i in range(10)]
W = [[v / sum(r) for v in r] for r in W]
X = [[1.0, v] for v in (0.1, 0.3, 0.5, 0.9, 0.2, 0.4, 0.8, 0.6, 0.7, 0.05)]
Y = [0, 1, 3, 5, 0, 2, 4, 0, 3, 0]


def test_is_sar_zip():
    r, ref = spatial_zip(Y, X, W, rho_bounds=(-0.5, 0.5)), sar_zip(Y, X, W, rho_bounds=(-0.5, 0.5))
    assert r.rho == ref.rho and r.loglik == ref.loglik and r.count_coefficients == ref.count_coefficients


def test_zero_model_and_aliases():
    r = spatial_zip(Y, X, W, rho_bounds=(0, 0))
    assert r.rho == 0 and len(r.zero_coefficients) == 1 and r.k == 4
    assert spat is spatial_zip and spatialzip is spatial_zip
