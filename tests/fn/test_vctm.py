"""Tests for morie.fn.vctm -- Victimization survey rate."""

import pytest

from morie.fn._containers import CrimeResult
from morie.fn.vctm import vctm, victimization_rate


class TestVctm:
    def test_alias(self):
        assert vctm is victimization_rate

    def test_basic_rate(self):
        result = victimization_rate(50, 1000)
        assert isinstance(result, CrimeResult)
        assert result.rate == pytest.approx(50.0)

    def test_ci_bounds(self):
        result = victimization_rate(50, 1000)
        assert result.ci_lower <= result.rate <= result.ci_upper
