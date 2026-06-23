"""Tests for morie.fn.mtotm — temporal pattern."""

import pandas as pd
import pytest

from morie.fn._containers import DescriptiveResult
from morie.fn.mtotm import mto_temporal_pattern


class TestTemporalPattern:
    def test_basic(self):
        dates = pd.date_range("2023-01-01", periods=100, freq="3h")
        r = mto_temporal_pattern(dates)
        assert isinstance(r, DescriptiveResult)
        assert "by_hour" in r.extra

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            mto_temporal_pattern([])
