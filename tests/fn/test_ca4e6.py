"""Tests for ca4e6.ca_chapter_4_equation_6."""

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

from morie.fn.ca4e6 import ca_chapter_4_equation_6


def test_ca4e6_basic():
    """Test basic functionality with a single scalar logit value."""
    rng = np.random.default_rng(42)
    logit = float(rng.normal(0, 1))
    result = ca_chapter_4_equation_6(logit)

    assert isinstance(result, dict)
    assert "value" in result
    assert "method" in result

    # Independent computation via the documented formula.
    expected = np.exp(logit) / (1.0 + np.exp(logit))
    assert abs(result["value"] - float(expected)) < 1e-12


def test_ca4e6_zero():
    """logit = 0 should map to probability = 0.5 exactly."""
    result = ca_chapter_4_equation_6(0.0)

    assert isinstance(result, dict)
    assert "value" in result
    expected = np.exp(0.0) / (1.0 + np.exp(0.0))
    assert abs(result["value"] - float(expected)) < 1e-12
    assert abs(result["value"] - 0.5) < 1e-12


def test_ca4e6_positive():
    """A positive logit should yield a probability > 0.5."""
    logit = 1.5
    result = ca_chapter_4_equation_6(logit)

    assert isinstance(result, dict)
    assert "value" in result
    expected = np.exp(logit) / (1.0 + np.exp(logit))
    assert abs(result["value"] - float(expected)) < 1e-12
    assert result["value"] > 0.5


def test_ca4e6_negative():
    """A negative logit should yield a probability < 0.5."""
    logit = -2.0
    result = ca_chapter_4_equation_6(logit)

    assert isinstance(result, dict)
    assert "value" in result
    expected = np.exp(logit) / (1.0 + np.exp(logit))
    assert abs(result["value"] - float(expected)) < 1e-12
    assert result["value"] < 0.5
    assert 0.0 < result["value"] < 1.0


def test_ca4e6_large_positive():
    """A very large positive logit should saturate near 1."""
    logit = 50.0
    result = ca_chapter_4_equation_6(logit)

    assert isinstance(result, dict)
    assert "value" in result
    expected = np.exp(logit) / (1.0 + np.exp(logit))
    assert abs(result["value"] - float(expected)) < 1e-12
    assert result["value"] > 0.999


def test_ca4e6_large_negative():
    """A very large negative logit should saturate near 0."""
    logit = -50.0
    result = ca_chapter_4_equation_6(logit)

    assert isinstance(result, dict)
    assert "value" in result
    expected = np.exp(logit) / (1.0 + np.exp(logit))
    assert abs(result["value"] - float(expected)) < 1e-12
    assert result["value"] < 0.001


def test_ca4e6_edge():
    """Test edge cases with a single scalar input."""
    logit = float(np.random.default_rng(42).normal(0, 1))
    result = ca_chapter_4_equation_6(logit)

    assert isinstance(result, dict)
    assert "value" in result
    expected = np.exp(logit) / (1.0 + np.exp(logit))
    assert abs(result["value"] - float(expected)) < 1e-12
