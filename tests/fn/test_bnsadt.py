"""Tests for bnsadt.bound_adversarial."""

from morie.fn import _array_core as np

from morie.fn.bnsadt import bound_adversarial


def test_bnsadt_basic():
    """Test basic functionality."""
    rng_y = np.random.default_rng(43)
    rng_d = np.random.default_rng(42)
    y = rng_y.normal(0, 1, 100)
    # Standard errors must be positive.
    D = rng_d.normal(0, 1, 100) ** 2 + 0.1
    family = "symmetric_step"
    result = bound_adversarial(y, D, family)
    assert isinstance(result, dict)
    # The function returns a RichResult that acts like a dict.
    assert "estimate" in result
    assert "uncorrected" in result
    assert "bound_lower" in result
    assert "bound_upper" in result
    assert "sweep" in result
    assert "mu" in result
    assert "tau" in result
    assert "betas" in result
    # The corrected estimate under the fitted selection model must
    # equal the naive point estimate plus the reported correction.
    assert result["estimate"] == result["uncorrected"] + result["correction"]
    # The adversarial interval must contain the corrected point
    # estimate when the swept selection probabilities span 1 down to
    # something below 1 (so selection is strictly stronger than none).
    assert result["bound_lower"] <= result["estimate"] <= result["bound_upper"]


def test_bnsadt_edge():
    """Test edge cases."""
    rng_y = np.random.default_rng(43)
    rng_d = np.random.default_rng(42)
    y = rng_y.normal(0, 1, 100)
    # Standard errors must be positive.
    D = rng_d.normal(0, 1, 100) ** 2 + 0.1
    family = "symmetric_step"
    result = bound_adversarial(y, D, family)
    assert isinstance(result, dict)
