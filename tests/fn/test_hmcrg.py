"""Tests for hmcrg.hierarchical_model."""

import math

import pytest

from morie.fn.hmcrg import hierarchical_model


Y = [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]    # BDA3 eight schools
S = [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]


def test_hmcrg_basic():
    """BDA3 5.4 at tau = 10: mu_hat and V_mu (5.20), theta_j | mu_hat, tau
    (5.17) with shrinkage sigma^2 / (sigma^2 + tau^2), and log p(tau | y)
    (5.21, flat prior), recomputed here."""
    tau = 10.0
    r = hierarchical_model(Y, S, tau)
    w = [1 / (s * s + tau * tau) for s in S]
    vmu = 1 / sum(w)
    mu = vmu * sum(a * b for a, b in zip(w, Y))
    assert (r["mu_hat"], r["V_mu"]) == pytest.approx((mu, vmu), rel=1e-14)
    for j in range(8):
        prec = 1 / S[j] ** 2 + 1 / tau ** 2
        assert r["theta_hat"][j] == pytest.approx((Y[j] / S[j] ** 2 + mu / tau ** 2) / prec, rel=1e-13)
        assert r["shrinkage"][j] == pytest.approx(S[j] ** 2 / (S[j] ** 2 + tau ** 2), rel=1e-13)
    lp = 0.5 * math.log(vmu) - 0.5 * sum(math.log(s * s + tau * tau) + (y - mu) ** 2 / (s * s + tau * tau)
                                         for y, s in zip(Y, S))
    assert r["log_post_tau"] == pytest.approx(lp, rel=1e-13)


def test_hmcrg_edge():
    """tau = 0 pools completely: every theta is mu_hat, the precision-
    weighted mean (7.69 for the eight schools, BDA3 p. 120)."""
    r = hierarchical_model(Y, S, 0.0)
    assert r["theta_hat"] == [r["mu_hat"]] * 8
    assert round(r["mu_hat"], 2) == 7.69
    with pytest.raises(ValueError, match="positive"):
        hierarchical_model(Y, [0.0] * 8, 5.0)


