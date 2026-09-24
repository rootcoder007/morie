"""Verification tests for gh_c8_6.

Ghosal and van der Vaart (2017), sec. 8.2, the i.i.d. posterior contraction theorem.
"""

import math

import pytest

from morie.fn.gh_c8_6 import ghosal_iid_crt_thm


def test_the_contraction_rate_balances_entropy_against_the_sample_size():
    # Sec. 8.2: a rate eps_n is attained when the entropy is at most
    # n eps^2 and the prior gives at least exp(-C n eps^2) mass
    x = [0.3, -0.5, 0.8, 0.1, -0.2, 0.6, 0.4, -0.7]
    res = ghosal_iid_crt_thm(x)
    n = res["n"]
    eps = float(res["eps"])
    assert res["n_eps_squared"] == pytest.approx(n * eps * eps, rel=1e-10)
    assert res["entropy_budget"] == pytest.approx(n * eps * eps, rel=1e-10)
    assert res["metric"] == "Hellinger"


def test_a_larger_sample_admits_a_faster_rate():
    small = ghosal_iid_crt_thm([0.1] * 8 + [0.2] * 8)
    large = ghosal_iid_crt_thm([0.1] * 40 + [0.2] * 40)
    assert float(large["eps"]) < float(small["eps"])
