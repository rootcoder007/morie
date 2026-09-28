"""Tests for morie.fn.svvt2 -- 2D vote trading equilibrium"""

from morie.fn import _array_core as np
from morie.fn.svvt2 import vote_trade_2d


class TestVoteTrade2d:
    def test_basic(self):
        x = np.array([[3.0, -1.0], [-1.0, 3.0], [-3.0, -3.0]])
        result = vote_trade_2d(x)
        assert result.value == len(result.extra["trades"]) == 1
        assert result.extra["welfare_after"] == -2.0

    def test_output_type(self):
        result = vote_trade_2d(np.array([[1.0, 2.0]]))
        assert hasattr(result, "value")
