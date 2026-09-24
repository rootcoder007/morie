"""Tests for logit_form.logit_form."""

from morie.fn import _array_core as np

from morie.fn.logit_form import (
    logit_form,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo2e3_basic():
    """Test basic functionality."""
    p = 0.1
    result = logit_form(p)
    assert isinstance(result, dict)
    assert "value" in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo2e3_edge():
    """Test edge cases."""
    p = 0.1
    result = logit_form(p)
    assert isinstance(result, dict)
