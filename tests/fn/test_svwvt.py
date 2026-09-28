"""Tests for morie.fn.svwvt -- Weighted voting game value"""

from morie.fn import _array_core as np

from morie.fn.svwvt import weighted_vote


class TestWeightedVote:
    def test_basic(self):
        x = np.array([1.0, 2.0])
        result = weighted_vote(x)
        assert result.value == [0.0, 1.0]  # quota 1.5: player 2 is a dictator

    def test_output_type(self):
        result = weighted_vote(np.array([1.0, 2.0]))
        assert hasattr(result, "value")
