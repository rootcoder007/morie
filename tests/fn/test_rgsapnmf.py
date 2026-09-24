"""Tests for rgsapnmf.rangayyan_sleep_apnea_nmf."""

from morie.fn import _array_core as np

from morie.fn.bsaclass import rangayyan_sleep_apnea_nmf


def test_rgsapnmf_basic():
    """Test basic functionality."""
    airflow = np.random.default_rng(42).normal(0.0, 1.0, 40)
    fs = 0.1
    result = rangayyan_sleep_apnea_nmf(airflow, fs)
    assert isinstance(result, dict)
    assert "ahi" in result


def test_rgsapnmf_edge():
    """Test edge cases."""
    airflow = np.random.default_rng(42).normal(0.0, 1.0, 40)
    fs = 0.1
    result = rangayyan_sleep_apnea_nmf(airflow, fs)
    assert isinstance(result, dict)
