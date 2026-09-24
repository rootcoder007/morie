"""Verification tests for gh_c14_4.

Ghosal and van der Vaart (2017), sec. 14.1.2, the Chinese restaurant franchise.
"""

import math

import pytest

from morie.fn.gh_c14_4 import ghosal_crf_def


def test_the_franchise_shares_dishes_across_restaurants():
    # sec. 14.1.2: new tables draw dishes from a global CRP, so the
    # restaurants share a dish menu
    res = ghosal_crf_def(n_per_rest=(40, 40), alpha=2.0, seed=6)
    assert res["dishes_shared"] is True
    assert res["n_global_dishes"] >= 1


def test_the_dish_count_grows_with_the_concentration():
    small = ghosal_crf_def(n_per_rest=(60, 60), alpha=0.5, seed=8)["n_global_dishes"]
    large = ghosal_crf_def(n_per_rest=(60, 60), alpha=20.0, seed=8)["n_global_dishes"]
    assert large >= small
