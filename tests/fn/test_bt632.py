"""Tests for bt632.boot_632_estimator."""

from morie.fn import _array_core as np

from morie.fn.bt632 import boot_632_estimator


def test_bt632_basic():
    """Test basic functionality."""
    # err_app and err_oob are scalar error values (rates)
    err_app = 0.25
    err_oob = 0.30
    result = boot_632_estimator(err_app, err_oob)
    assert isinstance(result, dict)
    assert "alias_of" in result
    assert result["alias_of"] == "morie.fn.eslo63.esl_oob_632"


def test_bt632_edge():
    """Test edge cases with equal errors."""
    err_app = 0.10
    err_oob = 0.10
    result = boot_632_estimator(err_app, err_oob)
    assert isinstance(result, dict)
    assert "alias_of" in result
