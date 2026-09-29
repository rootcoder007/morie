"""Tests for morie.fn.bayes_general: values recomputed from first principles."""

from morie.fn.bayes_general import bayes_general


def test_posterior_vector():
    pri, lik = [0.5, 0.3, 0.2], [0.1, 0.4, 0.8]
    pz = sum(a * b for a, b in zip(pri, lik))
    r = bayes_general(pri, lik)
    assert abs(r["p_z"] - pz) < 1e-15
    for got, a, b in zip(r["posteriors"], pri, lik):
        assert abs(got - a * b / pz) < 1e-15
    assert abs(sum(r["posteriors"]) - 1.0) < 1e-15
