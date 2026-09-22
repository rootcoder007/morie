"""Tests for fzb4x.fauzi_b4_coefficient."""

from morie.fn import _array_core as np

from morie.fn.fzb4x import fauzi_b4_coefficient


def test_fzb4x_basic():
    """Test basic functionality."""
    # fppp is the third derivative f_X^(3)(x) (a scalar),
    # and mu4 defaults to 3 (the Gaussian kernel value).
    fppp = 0.4
    result = fauzi_b4_coefficient(fppp)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "mu4" in result
    assert "method" in result
    # Independent recomputation from the documented formula
    # b_4 = fppp / 24 * mu4
    expected_estimate = float(fppp) / 24.0 * 3.0
    assert result["estimate"] == expected_estimate
    assert result["mu4"] == 3.0


def test_fzb4x_edge():
    """Test edge cases: custom mu4 and zero third derivative."""
    fppp = 0.0
    mu4 = 7.5
    result = fauzi_b4_coefficient(fppp, mu4=mu4)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "mu4" in result
    assert "method" in result
    # When fppp == 0, the bias coefficient is zero regardless of mu4.
    assert result["estimate"] == 0.0
    assert result["mu4"] == mu4

    # Non-zero fppp with custom mu4, recomputed independently.
    fppp2 = -2.4
    mu4_2 = 1.25
    result2 = fauzi_b4_coefficient(fppp2, mu4=mu4_2)
    expected_estimate2 = float(fppp2) / 24.0 * float(mu4_2)
    assert result2["estimate"] == expected_estimate2
    assert result2["mu4"] == mu4_2
