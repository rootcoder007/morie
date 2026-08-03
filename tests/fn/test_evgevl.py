"""Tests for evgevl.evt_gev_loglik."""
import math

from morie.fn.evgevl import evt_gev_loglik


def test_gumbel_closed_form():
    # xi = 0: ll = -n log s - sum t - sum exp(-t) (Coles eq. 3.9)
    x = [0.5, 1.0, 2.0]
    mu, s = 1.0, 2.0
    want = sum(-math.log(s) - (v - mu) / s - math.exp(-(v - mu) / s)
               for v in x)
    assert abs(evt_gev_loglik(x, mu, s, 0.0)["ll"] - want) < 1e-12


def test_outside_support_is_minus_inf():
    # xi = 1: support x > mu - sigma
    r = evt_gev_loglik([-5.0], 0.0, 1.0, 1.0)
    assert r["ll"] == float("-inf")
