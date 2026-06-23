"""Tests for morie.fn.jamste -- James-Stein shrinkage."""

from morie.fn._containers import DescriptiveResult
from morie.fn.jamste import james_stein, jamste


class TestJamste:
    def test_alias(self):
        assert jamste is james_stein

    def test_shrinks(self):
        x = [10.0, -5.0, 3.0, 0.1, -2.0]
        result = james_stein(x)
        assert isinstance(result, DescriptiveResult)
        assert 0 <= result.value <= 1

    def test_needs_three(self):
        import pytest

        with pytest.raises(ValueError, match="requires >= 3"):
            james_stein([1.0, 2.0])
