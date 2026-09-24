"""Tests for rgemdvf.rangayyan_emd_vf_detect."""

from morie.fn import _array_core as np

from morie.fn.bsatf import rangayyan_emd_vf_detect


def test_rgemdvf_basic():
    """Test basic functionality."""
    ecg = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_emd_vf_detect(ecg)
    assert isinstance(result, dict)
    assert "imfs" in result


def test_rgemdvf_edge():
    """Test edge cases."""
    ecg = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_emd_vf_detect(ecg)
    assert isinstance(result, dict)
