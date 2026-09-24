"""Verification tests for gh_c14_22.

Ghosal and van der Vaart (2017), sec. 14.9.5, the nested Dirichlet process.
"""

import math

import pytest

from morie.fn.gh_c14_22 import ghosal_nested_dp


def test_the_nested_process_clusters_groups_onto_shared_distributions():
    # sec. 14.9.5: G0 is itself atomic on distributions, so distinct
    # groups reuse the same G with positive probability
    res = ghosal_nested_dp(n_groups=8, gamma=1.0, alpha=1.0, seed=7)
    labels = list(res["group_labels"])
    assert len(labels) == 8
    assert len(set(labels)) < len(labels)
    assert res["groups_share_distributions"] is True


def test_a_single_group_cannot_share_with_anyone():
    res = ghosal_nested_dp(n_groups=1, seed=7)
    assert len(list(res["group_labels"])) == 1
