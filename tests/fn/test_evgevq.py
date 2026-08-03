"""Tests for evgevq.evt_gev_quantile."""
from morie.fn.evgevc import evt_gev_cdf
from morie.fn.evgevq import evt_gev_quantile


def test_roundtrip_all_shapes():
    for xi in (-0.3, 0.0, 0.2, 0.7):
        for p in (0.05, 0.5, 0.9, 0.99):
            x = evt_gev_quantile(p, 1.0, 2.0, xi)["x_p"]
            assert abs(evt_gev_cdf(x, 1.0, 2.0, xi)["F"] - p) < 1e-10
