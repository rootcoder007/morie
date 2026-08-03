"""Tests for gxe_blup_model.gxe_blup_model."""

from morie.fn import _array_core as np

from morie.fn.gxe_blup_model import gxe_blup_model


def test_msm018_basic():
    """Test basic functionality."""
    marker = np.random.default_rng(42).normal(0, 1, 100)
    information = np.random.default_rng(42).normal(0, 1, 100)
    prediction = np.random.default_rng(42).normal(0, 1, 100)
    although = np.random.default_rng(42).normal(0, 1, 100)
    this = np.random.default_rng(42).normal(0, 1, 100)
    could = np.random.default_rng(42).normal(0, 1, 100)
    result = gxe_blup_model(marker, information, prediction, although, this, could)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_msm018_edge():
    """Test edge cases."""
    marker = np.random.default_rng(42).normal(0, 1, 100)
    information = np.random.default_rng(42).normal(0, 1, 100)
    prediction = np.random.default_rng(42).normal(0, 1, 100)
    although = np.random.default_rng(42).normal(0, 1, 100)
    this = np.random.default_rng(42).normal(0, 1, 100)
    could = np.random.default_rng(42).normal(0, 1, 100)
    result = gxe_blup_model(marker, information, prediction, although, this, could)
    assert isinstance(result, dict)
