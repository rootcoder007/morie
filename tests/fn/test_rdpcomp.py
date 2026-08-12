"""Tests for rdpcomp (Renyi DP of the sampled Gaussian).

Replaces the generated stub, which imported a name the module never had.
"""

import math

from morie.fn.rdpcomp import rdp_compose, rdp_sampled_gaussian, rdpcomp


def test_full_sampling_is_the_plain_gaussian_mechanism():
    # q = 1 gives the closed form alpha / (2 sigma^2)
    for alpha in (2, 5, 32):
        for sigma in (0.8, 2.0):
            want = alpha / (2.0 * sigma * sigma)
            assert abs(rdp_sampled_gaussian(alpha, 1.0, sigma) -
                       want) < 1e-9


def test_subsampling_only_helps():
    for alpha in (2, 8):
        full = rdp_sampled_gaussian(alpha, 1.0, 1.0)
        part = rdp_sampled_gaussian(alpha, 0.01, 1.0)
        assert part < full


def test_more_noise_gives_less_leakage():
    quiet = rdp_sampled_gaussian(4, 0.1, 4.0)
    loud = rdp_sampled_gaussian(4, 0.1, 0.5)
    assert quiet < loud


def test_composition_is_additive():
    # Proposition 1: k identical mechanisms add their RDP
    one = rdp_sampled_gaussian(8, 0.05, 1.5)
    assert abs(rdp_compose(8, 0.05, 1.5, steps=10) - 10.0 * one) < 1e-9


def test_the_curve_grows_with_the_steps():
    a = rdpcomp(0.05, 1.5, alpha=[2, 4, 8], steps=1)["rdp_epsilons"]
    b = rdpcomp(0.05, 1.5, alpha=[2, 4, 8], steps=20)["rdp_epsilons"]
    assert all(b[i] > a[i] for i in range(3))


def test_conversion_to_epsilon_delta_is_reported():
    res = rdpcomp(0.01, 2.0, steps=100, delta=1e-5)
    assert res["estimate"] > 0
    assert len(res["alphas"]) == len(res["rdp_epsilons"])


def test_validation():
    for call in (lambda: rdp_sampled_gaussian(2.5, 0.1, 1.0),
                 lambda: rdp_sampled_gaussian(1, 0.1, 1.0),
                 lambda: rdp_sampled_gaussian(2, 0.0, 1.0),
                 lambda: rdp_sampled_gaussian(2, 1.5, 1.0),
                 lambda: rdp_sampled_gaussian(2, 0.1, 0.0)):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
