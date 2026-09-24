"""Tests for ca7e14.ca_chapter_7_equation_14."""
import math

from morie.fn import _array_core as np

from morie.fn.ca7e14 import ca_chapter_7_equation_14


def test_ca7e14_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    beta_0j = float(rng.normal(0, 1))
    beta_1j = float(rng.normal(0, 1))
    x_1ij = float(rng.normal(0, 1))
    result = ca_chapter_7_equation_14(beta_0j, beta_1j, x_1ij)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
    expected = beta_0j + beta_1j * x_1ij
    assert math.isclose(result["value"], expected)


def test_ca7e14_edge():
    """Test edge cases."""
    result = ca_chapter_7_equation_14(0.0, 0.0, 0.0)
    assert isinstance(result, dict)
    assert "value" in result
    assert result["value"] == 0.0
    assert math.isfinite(result["value"])
