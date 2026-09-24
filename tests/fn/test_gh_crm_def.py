"""Verification tests for gh_crm_def.

Ghosal and van der Vaart (2017), App. J, completely random measures.
"""

import math

import pytest

from morie.fn.gh_crm_def import ghosal_completely_random_measure


def test_a_completely_random_measure_is_independent_on_disjoint_sets():
    # App. J: M(A) and M(B) are independent for disjoint A and B, so
    # the empirical correlation across halves is near zero
    res = ghosal_completely_random_measure(n_sim=4000, seed=42)
    assert abs(float(res["estimate"])) < 0.1
    assert res["independent"] is True


def test_the_independence_check_is_stable_across_seeds():
    for seed in (1, 2, 3):
        res = ghosal_completely_random_measure(n_sim=2000, seed=seed)
        assert abs(float(res["estimate"])) < 0.15
