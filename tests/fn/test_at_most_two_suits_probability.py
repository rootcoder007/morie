"""Tests for morie.fn.at_most_two_suits_probability: values recomputed from first principles."""

import math

from morie.fn.at_most_two_suits_probability import at_most_two_suits_probability


def test_count_by_enumerating_suit_patterns():
    # hands with at most two suits: choose the suits used, count hands using exactly those
    one = 4 * math.comb(13, 5)
    two = math.comb(4, 2) * (math.comb(26, 5) - 2 * math.comb(13, 5))
    r = at_most_two_suits_probability()
    assert r["favorable"] == one + two
    assert r["total"] == math.comb(52, 5)
    assert abs(r["probability"] - (one + two) / math.comb(52, 5)) < 1e-15
