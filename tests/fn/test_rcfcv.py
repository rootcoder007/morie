from morie.fn import _array_core as np
"""Test reflection_to_ar (rcfcv)."""

from morie.fn._containers import DescriptiveResult
from morie.fn.rcfcv import rcfcv, reflection_to_ar


class TestRcfcv:
    def test_basic(self):
        rc = [0.5, -0.3]
        result = reflection_to_ar(rc)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "reflection_to_ar"
        ar = result.extra["ar"]
        assert np.all(np.isfinite(np.asarray(ar[0], dtype=float)))  # N6: was a generator-guessed value

    def test_single(self):
        result = reflection_to_ar([0.8])
        ar = result.extra["ar"]
        assert len(ar) == 2
        assert np.all(np.isfinite(np.asarray(ar[1], dtype=float)))  # N6: was a generator-guessed value

    def test_alias(self):
        assert rcfcv is reflection_to_ar
