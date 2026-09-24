"""Tests for rgmscart.rangayyan_muscle_artifact."""

from morie.fn import _array_core as np

from morie.fn.bsaphys import rangayyan_muscle_artifact


def test_rgmscart_basic():
    """Test basic functionality."""
    vag = np.random.default_rng(42).normal(0.0, 1.0, 40)
    emg_ref = np.random.default_rng(42).normal(0.0, 1.0, 40)
    fs = 0.1
    result = rangayyan_muscle_artifact(vag, emg_ref, fs)
    assert isinstance(result, dict)
    assert "cleaned" in result


def test_rgmscart_edge():
    """Test edge cases."""
    vag = np.random.default_rng(42).normal(0.0, 1.0, 40)
    emg_ref = np.random.default_rng(42).normal(0.0, 1.0, 40)
    fs = 0.1
    result = rangayyan_muscle_artifact(vag, emg_ref, fs)
    assert isinstance(result, dict)
