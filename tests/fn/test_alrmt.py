"""Tests for alrmt.alammar_reward_model_training_bt."""

from morie.fn.alrmt import alammar_reward_model_training_bt


def test_alrmt_basic():
    out = alammar_reward_model_training_bt([2.0], [0.0])
    assert out["pair_accuracy"] == 1.0


def test_alrmt_edge():
    import pytest
    with pytest.raises(ValueError, match="loser score"):
        alammar_reward_model_training_bt([1.0, 2.0], [0.0])
