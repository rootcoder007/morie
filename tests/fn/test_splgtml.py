"""Tests for morie.fn.splgtml: the Klier-McMillen GMM with an intercept added."""

from morie.fn.spdiscrete import spatial_logit_gmm
from morie.fn.splgtml import splgtml

N = 10
W = [[1.0 if abs(i - j) == 1 else 0.0 for j in range(N)] for i in range(N)]
W = [[v / sum(r) for v in r] for r in W]
V = (2.0, -1.0, 0.1, 1.5, 0.6, -0.4, 0.9, -1.3, 0.2, 1.1)
Y = [1, 0, 1, 1, 0, 0, 1, 0, 0, 1]


def test_intercept_and_rho():
    a = splgtml(Y, [[v] for v in V], W)
    b = spatial_logit_gmm(Y, [[1.0, v] for v in V], W)
    assert a["rho"] == b["rho"]
    assert list(a["coefficients"]) == list(b["coefficients"])
