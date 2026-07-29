"""Tests for alcbm.alammar_conversation_buffer_memory."""

from morie.fn.alcbm import alammar_conversation_buffer_memory


def test_alcbm_basic():
    out = alammar_conversation_buffer_memory([("u1", "a1"), ("u2", "a2")], 1)
    assert out["memory"] == [("u2", "a2")]
    assert out["turns_forgotten"] == 1


def test_alcbm_edge():
    import pytest
    with pytest.raises(ValueError, match="positive"):
        alammar_conversation_buffer_memory([], 0)
