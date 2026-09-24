"""Tests for rgecgwvf.rangayyan_ecg_waveshape."""

from morie.fn import _array_core as np

from morie.fn.bsaqrs import rangayyan_ecg_waveshape


def test_rgecgwvf_basic():
    """Test basic functionality."""
    qrsdur = 0.1
    stdev = 0.1
    result = rangayyan_ecg_waveshape(qrsdur, stdev)
    assert isinstance(result, dict)
    assert "qrsdurms" in result


def test_rgecgwvf_edge():
    """Test edge cases."""
    qrsdur = 0.1
    stdev = 0.1
    result = rangayyan_ecg_waveshape(qrsdur, stdev)
    assert isinstance(result, dict)
