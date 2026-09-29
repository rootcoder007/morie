"""Tests for morie.fn.conditional_from_joint: recompute Morin (2016) from the formula."""

from morie.fn.conditional_from_joint import conditional_from_joint


def test_ratio():
    for pab, pa in ((0.12, 0.3), (0.2, 0.5), (0.3, 0.3)):
        assert abs(conditional_from_joint(pab, pa)["p_b_given_a"] - pab / pa) < 1e-15
