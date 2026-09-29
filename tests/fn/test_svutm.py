"""Tests for morie.fn.svutm -- Spatial utility maximizer"""

from morie.fn import _array_core as np
from morie.fn.svutm import utility_max


class TestUtilityMax:
    def test_basic(self):
        x = np.array([1.0, 2.0])
        result = utility_max(x, ideal_point=np.array([0.0, 0.0]))
        assert result.value is not None
        assert result.value >= 0

    def test_output_type(self):
        result = utility_max(np.array([1.0, 2.0]))
        assert hasattr(result, "value")


def test_gaussian_utilities_recomputed():
    import math

    X = [[3.0, 0.0], [1.0, 1.0], [0.0, 2.0]]
    v = [0.5, 0.5]
    d2 = [sum((a - b) ** 2 for a, b in zip(r, v)) for r in X]
    r = utility_max(X, ideal_point=v, utility="gaussian", scale=2.0)
    assert r.extra["utilities"] == [math.exp(-d / 8.0) for d in d2]
    assert r.value == d2.index(min(d2))
