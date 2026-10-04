"""Tests for morie.fn.mtocyc — cyclist."""

import pytest

from morie.fn import _frame_core as pd
from morie.fn._containers import DescriptiveResult
from morie.fn.mtocyc import mto_cyclist


class TestCyclist:
    def test_basic(self):
        df = pd.DataFrame({"cyclist_involved": [1, 0, 0, 1], "severity": ["Minor", "Minor", "Serious", "Minor"]})
        r = mto_cyclist(df)
        assert isinstance(r, DescriptiveResult)
        assert r.value == pytest.approx(2.0)

    def test_missing_col(self):
        with pytest.raises(ValueError):
            mto_cyclist(pd.DataFrame({"x": [1]}))
