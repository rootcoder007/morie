"""Test kaiser_window (wnksr)."""

from morie.fn._containers import DescriptiveResult
from morie.fn.wnksr import kaiser_window, wnksr


class TestWnksr:
    def test_basic(self):
        result = kaiser_window(16, beta=5.0)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "kaiser_window"

    def test_length(self):
        result = kaiser_window(32)
        assert len(result.extra["window"]) == 32

    def test_alias(self):
        assert wnksr is kaiser_window
