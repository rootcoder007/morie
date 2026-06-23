"""Test multiclass_avg (mcavg)."""

import numpy as np

from morie.fn._containers import DescriptiveResult
from morie.fn.mcavg import mcavg, multiclass_avg


class TestMcavg:
    def test_perfect(self):
        y = np.array([0, 1, 2, 0, 1, 2])
        result = multiclass_avg(y, y, average="macro")
        assert isinstance(result, DescriptiveResult)
        assert result.extra["f1"] == 1.0

    def test_micro(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 2, 1, 0, 1, 2])
        result = multiclass_avg(y_true, y_pred, average="micro")
        assert 0 < result.extra["f1"] < 1

    def test_alias(self):
        assert mcavg is multiclass_avg
