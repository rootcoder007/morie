"""Tests for morie.fn.xrspl: spatial_logit redirects to the real estimator."""

from morie.fn.spdiscrete import spatial_logit_gmm
from morie.fn.xrspl import spat, spatial_logit, spatiallogit

W = [[1.0 if abs(i - j) == 1 else 0.0 for j in range(10)] for i in range(10)]
W = [[v / sum(r) for v in r] for r in W]
X = [[1.0, v] for v in (2.0, -1.0, 0.1, 1.5, 0.6, -0.4, 0.9, -1.3, 0.2, 1.1)]
Y = [1, 0, 1, 1, 0, 0, 1, 0, 0, 1]


def test_is_spatial_logit_gmm():
    r, ref = spatial_logit(Y, X, W), spatial_logit_gmm(Y, X, W)
    assert r.rho == ref.rho and r.coefficients == ref.coefficients and r.se == ref.se


def test_rho_zero_gives_logit_start():
    r = spatial_logit(Y, X, W)
    assert -1 < r.rho < 1 and len(r.coefficients) == 2


def test_aliases():
    assert spat is spatial_logit and spatiallogit is spatial_logit
