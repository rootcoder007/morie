"""Tests for mienco.mi_neural_encoder (local Deep InfoMax, Hjelm et al. 2019)."""

import math

import pytest

from morie.fn.mienco import mi_neural_encoder, prior_matching_loss


Y = [1.0, -0.5]
POS = [[0.8, -0.2], [1.2, -0.7], [0.3, 0.1]]
NEG = [[-0.4, 0.9], [0.2, 0.5]]


def crit(a, b):
    return sum(x * y for x, y in zip(a, b))


def sp(z):
    return math.log1p(math.exp(z))


def test_mienco_basic():
    """JSD bound E_P[-sp(-T)] - E_N[sp(T)] and DV bound
    E_P[T] - log E_N[e^T] over the local patch scores (eqs. 2-4)."""
    p = [crit(Y, v) for v in POS]
    q = [crit(Y, v) for v in NEG]
    r = mi_neural_encoder(Y, POS, NEG, crit)
    assert r["estimate"] == pytest.approx(sum(-sp(-v) for v in p) / 3 - sum(sp(v) for v in q) / 2, rel=1e-12)
    d = mi_neural_encoder(Y, POS, NEG, crit, estimator="dv")
    assert d["estimate"] == pytest.approx(sum(p) / 3 - math.log(sum(math.exp(v) for v in q) / 2), rel=1e-12)
    assert (r["n_patches"], r["n_negative_patches"]) == (3, 2)
    # prior matching (eq. 7): -[log D(prior) + log(1 - D(E(x)))] with D = sigmoid
    lo = prior_matching_loss([0.4, -1.0], [2.0], lambda z: z)
    sig = lambda z: 1 / (1 + math.exp(-z))
    assert lo == pytest.approx(-math.log(sig(2.0)) - (math.log(1 - sig(0.4)) + math.log(1 - sig(-1.0))) / 2, rel=1e-12)


def test_mienco_edge():
    """Unknown estimators and missing negatives raise."""
    with pytest.raises(ValueError):
        mi_neural_encoder(Y, POS, NEG, crit, estimator="nce")
    with pytest.raises(ValueError):
        mi_neural_encoder(Y, POS, [], crit)
