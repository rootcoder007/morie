"""Tests for rgstftp.rangayyan_stft_params."""

from morie.fn import _array_core as np

from morie.fn.bsatf import rangayyan_stft_params


def test_rgstftp_basic():
    """Test basic functionality."""
    fs = 0.1
    desired_t_res = 0.1
    desired_f_res = 0.1
    result = rangayyan_stft_params(fs, desired_t_res, desired_f_res)
    assert isinstance(result, dict)
    assert "nperseg_time" in result


def test_rgstftp_edge():
    """Test edge cases."""
    fs = 0.1
    desired_t_res = 0.1
    desired_f_res = 0.1
    result = rangayyan_stft_params(fs, desired_t_res, desired_f_res)
    assert isinstance(result, dict)
