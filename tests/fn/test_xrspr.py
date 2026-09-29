"""Tests for morie.fn.xrspr: spatial_panel_re redirects to the real estimator."""

from morie.fn.sppanel import spatial_panel_re_lag
from morie.fn.xrspr import spat, spatial_panel_re, spatialpanelre

W = [[0, 1, 0, 0], [0.5, 0, 0.5, 0], [0, 0.5, 0, 0.5], [0, 0, 1, 0]]
Y = [1.0, 2.0, 1.5, 0.3, 1.2, 2.4, 1.1, 0.8, 0.9, 2.2, 1.8, 0.1]
X = [[0.5], [1.0], [0.2], [0.3], [0.7], [1.3], [0.1], [0.6], [0.4], [0.9], [0.6], [0.2]]


def test_is_spatial_panel_re_lag():
    r, ref = spatial_panel_re(Y, X, W, 4), spatial_panel_re_lag(Y, X, W, 4)
    assert r.rho == ref.rho and r.phi == ref.phi and r.coefficients == ref.coefficients
    assert abs(r.sigma2_mu - (1 / (r.phi * r.phi) - 1) * r.sigma2 / 3) < 1e-12


def test_aliases():
    assert spat is spatial_panel_re and spatialpanelre is spatial_panel_re
