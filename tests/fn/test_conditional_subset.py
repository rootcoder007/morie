"""Tests for morie.fn.conditional_subset: recompute Morin (2016) from the formula."""

from morie.fn.conditional_subset import conditional_subset


def test_ratio():
    for pb, pa in ((0.1, 0.4), (0.0, 0.7), (0.35, 0.35)):
        assert abs(conditional_subset(pb, pa)["p_b_given_a"] - pb / pa) < 1e-15
