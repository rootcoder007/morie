"""Tests for morie.fn.suit_full_house_probability: values recomputed from first principles."""

import math

from morie.fn.suit_full_house_probability import suit_full_house_probability


def test_suit_pattern_count():
    fav = 4 * math.comb(13, 3) * 3 * math.comb(13, 2)
    r = suit_full_house_probability()
    assert r["favorable"] == fav
    assert abs(r["probability"] - fav / math.comb(52, 5)) < 1e-16
