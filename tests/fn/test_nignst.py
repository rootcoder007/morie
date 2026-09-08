"""Tests for nignst (normal-inverse-chi-square conjugate update).

Replaces the generated stub, which imported a name the module never had.
"""

from morie.fn.nignst import nignst


def test_the_posterior_update_is_the_textbook_one():
    y = [2.0, 4.0, 6.0, 8.0]
    mu0, k0, nu0, s0 = 1.0, 2.0, 3.0, 4.0
    res = nignst(y, mu0, k0, nu0, s0)
    n = len(y)
    ybar = sum(y) / n
    assert abs(res["ybar"] - ybar) < 1e-12
    assert abs(res["kappa_n"] - (k0 + n)) < 1e-12
    assert abs(res["nu_n"] - (nu0 + n)) < 1e-12
    assert abs(res["mu_n"] - (k0 * mu0 + n * ybar) / (k0 + n)) < 1e-12
    ss = sum((v - ybar) ** 2 for v in y)
    want = (nu0 * s0 + ss +
            k0 * n * (ybar - mu0) ** 2 / (k0 + n)) / (nu0 + n)
    assert abs(res["sigma_n_sq"] - want) < 1e-12


def test_a_vague_prior_leaves_the_sample_mean_standing():
    y = [1.0, 2.0, 3.0, 4.0, 5.0]
    res = nignst(y, 0.0, 1e-8, 1e-8, 1.0)
    assert abs(res["mu_n"] - 3.0) < 1e-6


def test_a_strong_prior_pulls_the_posterior_toward_it():
    y = [10.0] * 4
    weak = nignst(y, 0.0, 0.01, 1.0, 1.0)["mu_n"]
    strong = nignst(y, 0.0, 1000.0, 1.0, 1.0)["mu_n"]
    assert abs(strong) < abs(weak)
    assert 0.0 < strong < weak


def test_the_predictive_scale_exceeds_the_mean_scale():
    res = nignst([1.0, 2.0, 3.0], 0.0, 1.0, 2.0, 1.0)
    assert res["pred_scale_sq"] > res["mu_scale_sq"]


def test_more_data_shrinks_the_posterior_scale_of_the_mean():
    small = nignst([1.0, 2.0], 0.0, 1.0, 2.0, 1.0)["mu_scale_sq"]
    large = nignst([1.0, 2.0] * 50, 0.0, 1.0, 2.0, 1.0)["mu_scale_sq"]
    assert large < small


def test_validation():
    for call in (lambda: nignst([], 0.0, 1.0, 1.0, 1.0),
                 lambda: nignst([1.0], 0.0, 0.0, 1.0, 1.0),
                 lambda: nignst([1.0], 0.0, 1.0, -1.0, 1.0),
                 lambda: nignst([1.0], 0.0, 1.0, 1.0, 0.0)):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
