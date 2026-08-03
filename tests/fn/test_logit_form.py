"""Tests for logit_form.logit_form."""

from morie.fn import _array_core as np

from morie.fn.logit_form import (
    logit_form,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo2e3_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = logit_form(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo2e3_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = logit_form(x)
    assert isinstance(result, dict)
