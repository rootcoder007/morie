"""Tests for bezout.bezout."""

from morie.fn import _array_core as np

from morie.fn.bezout import bezout


def test_bezout_basic():
    """Test basic functionality with known identity a*x + b*y = gcd(a, b)."""
    a = 30
    b = 18
    result = bezout(a, b)

    assert isinstance(result, dict)
    assert "gcd" in result
    assert "x" in result
    assert "y" in result
    assert "check" in result

    # gcd(30, 18) = 6
    assert result["gcd"] == 6

    # The documented identity: a*x + b*y == gcd(a, b)
    expected_check = a * result["x"] + b * result["y"]
    assert expected_check == result["gcd"]
    assert result["check"] == expected_check


def test_bezout_edge():
    """Test edge cases: coprime inputs and one argument zero."""
    # Coprime integers: gcd should be 1
    a = 17
    b = 13
    result = bezout(a, b)

    assert isinstance(result, dict)
    assert result["gcd"] == 1

    expected_check = a * result["x"] + b * result["y"]
    assert expected_check == result["gcd"]
    assert result["check"] == expected_check

    # One argument zero: gcd(a, 0) == |a|, with x = sign(a), y = 0
    a = 24
    b = 0
    result = bezout(a, b)

    assert isinstance(result, dict)
    assert result["gcd"] == 24

    expected_check = a * result["x"] + b * result["y"]
    assert expected_check == result["gcd"]
    assert result["check"] == expected_check
