"""Tests for morie.fn.xrspp: spatial_probit redirects to the real estimator."""

import pytest

from morie.fn.sarbayes import sar_probit_gibbs
from morie.fn.spdiscrete import spatial_probit_gmm
from morie.fn.xrspp import spat, spatial_probit, spatialprobit

W = [[1.0 if abs(i - j) == 1 else 0.0 for j in range(10)] for i in range(10)]
W = [[v / sum(r) for v in r] for r in W]
X = [[1.0, v] for v in (2.0, -1.0, 0.1, 1.5, 0.6, -0.4, 0.9, -1.3, 0.2, 1.1)]
Y = [1, 0, 1, 1, 0, 0, 1, 0, 0, 1]


def test_gmm_is_spatial_probit_gmm():
    r, ref = spatial_probit(Y, X, W), spatial_probit_gmm(Y, X, W)
    assert r.rho == ref.rho and r.coefficients == ref.coefficients and r.se == ref.se


def test_bayes_is_sar_probit_gibbs():
    r = spatial_probit(Y, X, W, method="bayes", ndraw=30, burn_in=5, seed=3)
    ref = sar_probit_gibbs(Y, X, W, ndraw=30, burn_in=5, seed=3)
    assert r.rho_draws == ref.rho_draws and r.rho == ref.rho


def test_aliases_and_bad_method():
    assert spat is spatial_probit and spatialprobit is spatial_probit
    with pytest.raises(ValueError):
        spatial_probit(Y, X, W, method="default")
