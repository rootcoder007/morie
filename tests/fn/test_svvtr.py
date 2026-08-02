"""Tests for morie.fn.svvtr -- Vote trading (logrolling) model"""

from morie.fn import _array_core as np

from morie.fn.svvtr import vote_trading


class TestVoteTrading:
    def test_basic(self):
        x = np.array([1.0, 2.0])
        result = vote_trading(x, ideal_point=np.array([0.0, 0.0]))
        assert result.value is not None
        assert result.value >= 0

    def test_output_type(self):
        result = vote_trading(np.array([1.0, 2.0]))
        assert hasattr(result, "value")
