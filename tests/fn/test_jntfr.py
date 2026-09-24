"""Tests for jntfr.joint_frailty."""

from morie.fn import _array_core as np

from morie.fn.jntfr import joint_frailty


def test_jntfr_basic():
    """Test basic functionality."""
    time = 0.5
    event = 0.5
    terminal = 1
    cluster = 0.5
    result = joint_frailty(time, event, terminal, cluster)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_jntfr_edge():
    """Test edge cases."""
    time = 0.5
    event = 0.5
    terminal = 1
    cluster = 0.5
    result = joint_frailty(time, event, terminal, cluster)
    assert isinstance(result, dict)
