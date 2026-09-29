"""Tests for morie.fn.xrspn: spatial_panel_fe redirects to the real estimator."""

import pytest

from morie.fn.sppanel import spatial_panel_ml
from morie.fn.xrspn import spat, spatial_panel_fe, spatialpanelfe

W = [[0, 1, 0, 0], [0.5, 0, 0.5, 0], [0, 0.5, 0, 0.5], [0, 0, 1, 0]]
Y = [1.0, 2.0, 1.5, 0.3, 1.2, 2.4, 1.1, 0.8, 0.9, 2.2, 1.8, 0.1]
X = [[0.5], [1.0], [0.2], [0.3], [0.7], [1.3], [0.1], [0.6], [0.4], [0.9], [0.6], [0.2]]


def test_is_spatial_panel_ml():
    for model in ("lag", "error", "durbin"):
        for eff in ("individual", "time", "twoways"):
            r = spatial_panel_fe(Y, X, W, 4, model=model, effects=eff)
            ref = spatial_panel_ml(Y, X, W, 4, model=model, effects=eff)
            assert r.rho == ref.rho and r.coefficients == ref.coefficients


def test_pooled_rejected_and_aliases():
    with pytest.raises(ValueError):
        spatial_panel_fe(Y, X, W, 4, effects="pooled")
    assert spat is spatial_panel_fe and spatialpanelfe is spatial_panel_fe
