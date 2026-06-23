"""Tests for morie.fn.capam -- circular statistics."""

from morie.fn._containers import DescriptiveResult
from morie.fn.capam import capam, circular_mean


class TestCapam:
    def test_alias(self):
        assert capam is circular_mean

    def test_north(self):
        angles = [350, 10, 0, 5, 355]
        r = circular_mean(angles)
        assert isinstance(r, DescriptiveResult)
        mean_dir = r.value["mean_direction"]
        assert mean_dir < 20 or mean_dir > 340

    def test_high_concentration(self):
        angles = [90.0] * 20
        r = circular_mean(angles)
        assert r.value["mean_resultant_length"] > 0.99
        assert abs(r.value["mean_direction"] - 90) < 1
