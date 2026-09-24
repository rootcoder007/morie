"""Verification tests for gh_c13_6.

Ghosal and van der Vaart (2017), sec. 13.3.3, the beta-process jump path.
"""

import math

import pytest

from morie.fn.gh_c13_6 import ghosal_bp_path_gen


def test_the_beta_process_path_is_pure_jump_and_non_decreasing():
    # sec. 13.3.3: H(t) = sum_{tau_k <= t} J_k over a Poisson process
    res = ghosal_bp_path_gen(c=1.5, t_max=1.0, n_jumps=200, seed=2)
    assert res["pure_jump_nondecreasing"] is True
    assert res["estimate"] > 0.0


def test_more_jumps_accumulate_more_mass():
    few = ghosal_bp_path_gen(c=1.5, t_max=1.0, n_jumps=50, seed=5)["estimate"]
    many = ghosal_bp_path_gen(c=1.5, t_max=1.0, n_jumps=400, seed=5)["estimate"]
    assert many >= few
