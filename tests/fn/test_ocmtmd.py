"""Tests for ocmtmd.outcome_model_diagnostic."""

from morie.fn import _array_core as np

from morie.fn.ocmtmd import outcome_model_diagnostic


def test_ocmtmd_basic():
    """Test basic functionality."""
    y = 0.5
    A = 1
    H = 0.5
    result = outcome_model_diagnostic(y, A, H)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_ocmtmd_edge():
    """Test edge cases."""
    y = 0.5
    A = 1
    H = 0.5
    result = outcome_model_diagnostic(y, A, H)
    assert isinstance(result, dict)
