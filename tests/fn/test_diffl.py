"""Tests for diffl -- DIF flag summary."""

import pandas as pd

from morie.fn._containers import DescriptiveResult, DIFResult
from morie.fn.diffl import dif_flag_summary


class TestDifFlagSummary:
    def test_empty(self):
        result = dif_flag_summary()
        assert isinstance(result, DescriptiveResult)

    def test_two_methods(self):
        r1 = DIFResult(method="MH", items=pd.DataFrame({"item": ["i1", "i2"]}), flagged=["i1"])
        r2 = DIFResult(method="LR", items=pd.DataFrame({"item": ["i1", "i2"]}), flagged=["i1", "i2"])
        result = dif_flag_summary(r1, r2)
        assert result.extra["n_methods"] == 2
