"""Tests for morie.fn.vctrs -- weight initialization."""

from morie.fn._containers import DescriptiveResult
from morie.fn.vctrs import vctrs, weight_init


class TestVctrs:
    def test_alias(self):
        assert vctrs is weight_init

    def test_xavier_shape(self):
        r = weight_init(100, 50, method="xavier_uniform")
        assert isinstance(r, DescriptiveResult)
        assert r.value.shape == (100, 50)

    def test_he_variance(self):
        r = weight_init(1000, 500, method="he_normal")
        expected = 2.0 / 1000
        assert abs(r.extra["variance"] - expected) < 0.005
