"""Tests for gblup_model.gblup_model."""

from morie.fn import _array_core as np

from morie.fn.gblup_model import gblup_model


def test_msm015_basic():
    """Test basic functionality."""
    derived = np.random.default_rng(42).normal(0, 1, 100)
    re = np.random.default_rng(42).normal(0, 1, 100)
    ectance = np.random.default_rng(42).normal(0, 1, 100)
    information = np.random.default_rng(42).normal(0, 1, 100)
    Krause = np.random.default_rng(42).normal(0, 1, 100)
    et = np.random.default_rng(42).normal(0, 1, 100)
    result = gblup_model(derived, re, ectance, information, Krause, et)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_msm015_edge():
    """Test edge cases."""
    derived = np.random.default_rng(42).normal(0, 1, 100)
    re = np.random.default_rng(42).normal(0, 1, 100)
    ectance = np.random.default_rng(42).normal(0, 1, 100)
    information = np.random.default_rng(42).normal(0, 1, 100)
    Krause = np.random.default_rng(42).normal(0, 1, 100)
    et = np.random.default_rng(42).normal(0, 1, 100)
    result = gblup_model(derived, re, ectance, information, Krause, et)
    assert isinstance(result, dict)
