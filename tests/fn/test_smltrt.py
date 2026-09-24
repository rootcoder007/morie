"""Tests for smltrt.survey_ratio."""

from morie.fn import _array_core as np

from morie.fn.smltrt import survey_ratio


def test_smltrt_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = survey_ratio(y, x)
    assert isinstance(result, dict)
    assert "ratio" in result


def test_smltrt_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = survey_ratio(y, x)
    assert isinstance(result, dict)
