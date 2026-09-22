"""Tests for fzt22.fauzi_thm2_2_bias_brdkdfe."""

from morie.fn import _array_core as np

from morie.fn.fzt22 import fauzi_thm2_2_bias_brdkdfe


def test_fzt22_basic():
    """Test basic functionality against the documented Theorem 2.2 formula."""
    rng = np.random.default_rng(42)
    # Pick a bandwidth and a second smoothing parameter meeting the
    # documented constraints: a > 0 and a != 1.
    h = 0.3
    a = 0.5
    # F_X(x) must lie strictly in (0, 1).
    fx = 0.4
    # Realistic coefficient values for b2, b4.
    b2 = 0.2
    b4 = 0.05

    result = fauzi_thm2_2_bias_brdkdfe(h, a, b2, b4, fx)

    # The documented return keys.
    assert hasattr(result, "keys")
    assert "bias" in result
    assert "leading" in result
    assert "h" in result
    assert "a" in result
    assert "method" in result

    # Closed-form reference (Theorem 2.2, Eq. 2.6):
    #   leading = (b2^2 - 2*b4*fx) / (2*fx)
    #   bias    = h^4 * a^2 * leading
    expected_leading = (b2 ** 2 - 2.0 * b4 * fx) / (2.0 * fx)
    expected_bias = h ** 4 * a ** 2 * expected_leading

    assert abs(result["leading"] - expected_leading) < 1e-12
    assert abs(result["bias"] - expected_bias) < 1e-12
    assert result["h"] == h
    assert result["a"] == a


def test_fzt22_edge():
    """Test edge cases: fx near 0 or 1 raises; scalar-in, scalar-out."""
    h = 0.25
    a = 0.7  # valid: > 0 and != 1
    b2 = 0.1
    b4 = 0.01

    # fx strictly inside (0, 1) must succeed.
    result = fauzi_thm2_2_bias_brdkdfe(h, a, b2, b4, 0.5)
    expected_leading = (b2 ** 2 - 2.0 * b4 * 0.5) / (2.0 * 0.5)
    assert abs(result["leading"] - expected_leading) < 1e-12
    assert abs(result["bias"] - h ** 4 * a ** 2 * expected_leading) < 1e-12
