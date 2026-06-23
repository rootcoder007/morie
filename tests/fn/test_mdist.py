"""Test model_distance (mdist)."""

from morie.fn._containers import DescriptiveResult
from morie.fn.mdist import mdist, model_distance


class TestMdist:
    def test_identical(self):
        ar = [1.0, -0.5, 0.2]
        result = model_distance(ar, ar)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "model_distance"
        assert abs(result.value) < 1e-10

    def test_different(self):
        ar1 = [1.0, -0.5]
        ar2 = [1.0, -0.9]
        result = model_distance(ar1, ar2)
        assert result.value > 0

    def test_alias(self):
        assert mdist is model_distance
