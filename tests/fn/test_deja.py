"""Tests for morie.fn.deja -- near-duplicate detection."""

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

from morie.fn._containers import DescriptiveResult
from morie.fn.deja import deja, detect_duplicates


class TestDeja:
    def test_alias(self):
        assert deja is detect_duplicates

    def test_exact_duplicates(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
        result = detect_duplicates(df, threshold=0.99)
        assert isinstance(result, DescriptiveResult)
        assert result.value >= 1

    def test_no_duplicates(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"a": rng.normal(0, 10, 20), "b": rng.normal(0, 10, 20)})
        result = detect_duplicates(df, threshold=0.99999)
        assert result.value == 0
