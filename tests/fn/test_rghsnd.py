"""Tests for rghsnd.rangayyan_heart_sound_id."""

from morie.fn import _array_core as np

from morie.fn.bsaqrs import rangayyan_heart_sound_id


def test_rghsnd_basic():
    """Test basic functionality."""
    ecg = np.random.default_rng(42).normal(0.0, 1.0, 40)
    cp = np.random.default_rng(42).normal(0.0, 1.0, 40)
    fs = 0.1
    result = rangayyan_heart_sound_id(ecg, cp, fs)
    assert isinstance(result, dict)
    assert "s1" in result


def test_rghsnd_edge():
    """Test edge cases."""
    ecg = np.random.default_rng(42).normal(0.0, 1.0, 40)
    cp = np.random.default_rng(42).normal(0.0, 1.0, 40)
    fs = 0.1
    result = rangayyan_heart_sound_id(ecg, cp, fs)
    assert isinstance(result, dict)
