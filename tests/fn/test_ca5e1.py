"""Tests for ca5e1.ca_chapter_5_equation_1."""

from morie.fn import _array_core as np

from morie.fn.ca5e1 import ca_chapter_5_equation_1


def test_ca5e1_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    xs = rng.normal(0, 1, 100)
    bs = rng.normal(0, 1, 100)
    b0 = 0.5
    result = ca_chapter_5_equation_1(b0, bs, xs)
    assert isinstance(result, dict)
    assert "value" in result
    # Verify formula: logit(y=1|x) = b0 + sum(b_i * x_i)
    expected = b0 + (bs * xs).sum()
    assert abs(result["value"] - expected) < 1e-10


def test_ca5e1_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    xs = rng.normal(0, 1, 100)
    bs = rng.normal(0, 1, 100)
    b0 = 0.0
    result = ca_chapter_5_equation_1(b0, bs, xs)
    assert isinstance(result, dict)
    assert "value" in result
    # With b0=0: value = sum(b_i * x_i)
    expected = (bs * xs).sum()
    assert abs(result["value"] - expected) < 1e-10
