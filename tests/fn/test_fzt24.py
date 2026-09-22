"""Tests for fzt24.fauzi_thm2_4_mise_brdkdfe."""

from morie.fn import _array_core as np

from morie.fn.fzt24 import fauzi_thm2_4_mise_brdkdfe


def test_fzt24_basic():
    """Test basic functionality."""
    n = 100
    h = 0.3
    a = 2.0
    biasint = 0.5
    varint = 0.25
    result = fauzi_thm2_4_mise_brdkdfe(n, h, a, biasint, varint)
    assert isinstance(result, dict)
    # Key names per the docstring (RichResult with "mise", "biasterm",
    # "varterm", "smoothgain", "h", "a", "method").
    for key in ("mise", "biasterm", "varterm", "smoothgain", "h", "a", "method"):
        assert key in result

    # Formula from Theorem 2.4:
    #   MISE = h^8 a^4 biasint + varint/n - (h/n)[2(a^4+1)/(a^2-1)^2 r1 + r2]
    from morie.fn.fzr1 import kdfr1
    from morie.fn.fzr2 import kdfr2
    r1 = float(kdfr1()["estimate"])
    r2 = float(kdfr2(a=a)["estimate"])

    expected_bias = h ** 8 * a ** 4 * biasint
    expected_var = varint / n
    expected_gain = h / n * (2.0 * (a ** 4 + 1.0) / (a * a - 1.0) ** 2 * r1 + r2)
    expected_mise = expected_bias + expected_var - expected_gain

    assert result["biasterm"] == expected_bias
    assert result["varterm"] == expected_var
    assert result["smoothgain"] == expected_gain
    assert result["mise"] == expected_mise
    assert result["h"] == h
    assert result["a"] == a


def test_fzt24_edge():
    """Test edge cases."""
    n = 50
    h = 0.4
    a = 1.5
    biasint = 0.1
    varint = 0.2
    r1 = 0.123
    r2 = 0.456
    result = fauzi_thm2_4_mise_brdkdfe(n, h, a, biasint, varint, r1=r1, r2=r2)
    assert isinstance(result, dict)
    assert "mise" in result
    assert "biasterm" in result
    assert "varterm" in result
    assert "smoothgain" in result
