"""Tests for morie.fn.xrspc: spatial_poisson redirects to the real estimator."""

import math

from morie.fn.spcount import sar_poisson
from morie.fn.xrspc import spat, spatial_poisson, spatialpoisson

W = [[1.0 if abs(i - j) == 1 else 0.0 for j in range(10)] for i in range(10)]
W = [[v / sum(r) for v in r] for r in W]
X = [[1.0, v] for v in (0.1, 0.3, 0.5, 0.9, 0.2, 0.4, 0.8, 0.6, 0.7, 0.05)]
Y = [0, 1, 3, 5, 0, 2, 4, 0, 3, 0]


def test_is_sar_poisson():
    r, ref = spatial_poisson(Y, X, W), sar_poisson(Y, X, W)
    assert r.rho == ref.rho and r.loglik == ref.loglik and r.coefficients == ref.coefficients


def test_fitted_mean_formula():
    r = spatial_poisson(Y, X, W, rho_bounds=(0, 0))
    mu = [math.exp(r.coefficients[0] + r.coefficients[1] * x[1]) for x in X]
    assert max(abs(a - b) for a, b in zip(mu, r.fitted)) < 1e-12
    assert spat is spatial_poisson and spatialpoisson is spatial_poisson
