"""Verification tests for information_theory_mackay3e21.postodds.

The expected values are recomputed from MacKay (2003) eq. (3.21) p. 53 in the test body, so a
drift in the implementation fails the test.
"""

import math

import pytest

from morie.fn.information_theory_mackay3e21 import postodds


def test_postodds_is_the_likelihood_ratio_times_the_prior_odds():
    lik1, lik0, pr1, pr0 = 0.02, 0.005, 0.3, 0.7
    res = postodds(lik1, lik0, pr1, pr0)
    assert res["bayesfactor"] == pytest.approx(lik1 / lik0, rel=1e-12)
    assert res["odds"] == pytest.approx((lik1 / lik0) * (pr1 / pr0), rel=1e-12)
    assert res["p1"] == pytest.approx(res["odds"] / (1.0 + res["odds"]), rel=1e-12)


def test_postodds_with_equal_priors_reduces_to_the_bayes_factor():
    res = postodds(0.4, 0.1)
    assert res["odds"] == pytest.approx(res["bayesfactor"], rel=1e-12)
