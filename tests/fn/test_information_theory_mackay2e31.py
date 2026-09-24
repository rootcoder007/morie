"""Verification tests for information_theory_mackay2e31.urnpred.

The expected values are recomputed from MacKay (2003) eq. (2.29)-(2.31) p. 29 in the test body, so a
drift in the implementation fails the test.
"""

import math

import pytest

from morie.fn.information_theory_mackay2e31 import urnpred


def test_urnpred_is_the_posterior_mean_of_the_urn_fraction():
    nb, ntot, nurns = 3, 10, 10
    # (2.25): posterior over urns, uniform prior and binomial likelihood
    joint = [math.comb(ntot, nb) * (u / nurns) ** nb * (1.0 - u / nurns) ** (ntot - nb)
             for u in range(nurns + 1)]
    total = sum(joint)
    post = [j / total for j in joint]
    expected = sum((u / nurns) * post[u] for u in range(nurns + 1))
    res = urnpred(nb, ntot, nurns)
    assert res["p"] == pytest.approx(expected, abs=1e-12)
    assert res["pnot"] == pytest.approx(1.0 - expected, abs=1e-12)


def test_urnpred_with_no_data_is_one_half_by_symmetry():
    assert urnpred(0, 0)["p"] == pytest.approx(0.5, abs=1e-12)


def test_urnpred_lies_between_the_sample_proportion_and_one_half():
    res = urnpred(8, 10)
    assert 0.5 < res["p"] < 0.8
