"""Tests for morie.fn.siutrn — SIU trend."""

import pytest

from morie.fn.siutrn import siu_trend


class TestSiuTrend:
    def test_increasing(self):
        r = siu_trend([10, 15, 20, 25, 30])
        assert r.extra["trend"] == "increasing"

    def test_too_short(self):
        with pytest.raises(ValueError):
            siu_trend([5, 10])
