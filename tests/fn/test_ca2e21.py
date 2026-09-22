"""Tests for ca2e21.ca_chapter_2_equation_21."""

from morie.fn import _array_core as np

from morie.fn.ca2e21 import ca_chapter_2_equation_21


def test_ca2e21_basic():
    """Test basic functionality for the dummy = 0 subgroup regression equation."""
    rng = np.random.default_rng(42)
    b0 = 1.5
    b1 = 0.7
    b2 = -0.3
    bs = np.array([b0, b1, b2])
    dummy_index = 0
    result = ca_chapter_2_equation_21(b0, bs, dummy_index)
    assert isinstance(result, dict)
    assert "intercept" in result
    # Independent computation: y = b0 + b1*x1 + b2*x2 with the dummy term dropping out at 0
    # equals simply b0 (since the dummy contribution is b_index * 0 = 0).
    expected_intercept = b0
    assert result["intercept"] == expected_intercept
    assert result["intercept"] == b0
    assert "value" in result
    assert result["value"] == result["intercept"]
    assert result.get("method") == "Weisburd et al. (2022) eq. (2.21)"


def test_ca2e21_edge():
    """Test edge cases: bs supplied as a plain Python sequence."""
    b0 = -2.0
    b1 = 4.0
    b2 = 0.5
    bs = [b0, b1, b2]
    dummy_index = 0
    result = ca_chapter_2_equation_21(b0, bs, dummy_index)
    assert isinstance(result, dict)
    assert "intercept" in result
    # Computed independently from the documented formula with the dummy term dropping out.
    expected_intercept = b0
    assert result["intercept"] == expected_intercept
