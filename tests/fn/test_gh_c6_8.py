"""Verification tests for gh_c6_8.

Ghosal and van der Vaart (2017), Thm 6.27, consistency of tail-free priors.
"""

import math

import pytest

from morie.fn.gh_c6_8 import ghosal_tailfree_con


def test_tail_free_priors_are_consistent_so_the_error_shrinks():
    # Thm 6.27: the cell-probability posterior is consistent at every P0
    res = ghosal_tailfree_con(theta0=(0.1, 0.4, 0.3, 0.2), seed=11)
    errs = [float(v) for v in res["sup_error_by_n"]]
    assert res["improving"] is True
    assert errs[-1] < errs[0]
    assert all(v >= 0.0 for v in errs)


def test_the_error_is_a_supremum_over_a_probability_vector():
    res = ghosal_tailfree_con(theta0=(0.25, 0.25, 0.25, 0.25), seed=11)
    assert 0.0 <= float(res["estimate"]) <= 1.0
