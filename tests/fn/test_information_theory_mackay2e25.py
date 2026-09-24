"""Verification tests for information_theory_mackay2e25.urnpost.

The expected values are recomputed from MacKay (2003) eq. (2.25)-(2.26) p. 28 in the test body, so a
drift in the implementation fails the test.
"""

import math

import pytest

from morie.fn.information_theory_mackay2e25 import urnpost


def test_urnpost_matches_the_binomial_posterior_over_urns():
    nb, ntot, nurns = 3, 10, 10
    prior = 1.0 / (nurns + 1.0)
    joint = [prior * math.comb(ntot, nb) * (u / nurns) ** nb * (1.0 - u / nurns) ** (ntot - nb)
             for u in range(nurns + 1)]
    evid = sum(joint)
    res = urnpost(nb, ntot, nurns)
    assert res["evidence"] == pytest.approx(evid, rel=1e-12)
    for u in range(nurns + 1):
        assert res["posterior"][u] == pytest.approx(joint[u] / evid, abs=1e-12)
    assert res["map"] == 3


def test_urnpost_posterior_sums_to_one():
    res = urnpost(7, 20)
    assert sum(res["posterior"]) == pytest.approx(1.0, abs=1e-12)


def test_urnpost_rejects_more_black_balls_than_draws():
    with pytest.raises(ValueError):
        urnpost(5, 3)
