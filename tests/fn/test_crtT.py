"""Tests for crtT.chinese_remainder."""

from morie.fn import _array_core as np

from morie.fn.crtT import chinese_remainder


def test_crtT_basic():
    """Test basic functionality."""
    a = [2, 3, 2]
    m = [3, 5, 7]
    result = chinese_remainder(a, m)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "modulus" in result
    assert result["modulus"] == 3 * 5 * 7
    assert "residues" in result
    assert "moduli" in result
    # Independent computation of the least non-negative solution for
    # x = 2 (mod 3), x = 3 (mod 5), x = 2 (mod 7).
    mod = 3 * 5 * 7
    x = 23
    assert x % 3 == 2 and x % 5 == 3 and x % 7 == 2
    assert result["estimate"] == x % mod
    assert result["estimate"] < result["modulus"]
    assert all(r % mm == result["estimate"] % mm
               for r, mm in zip(result["residues"], result["moduli"]))


def test_crtT_edge():
    """Test edge cases."""
    # Two-congruence case: x = 4 (mod 7), x = 5 (mod 9).
    a = [4, 5]
    m = [7, 9]
    result = chinese_remainder(a, m)
    assert isinstance(result, dict)
    assert result["modulus"] == 7 * 9
    mod = 7 * 9
    # Brute-force search gives the canonical solution.
    x = next(k for k in range(mod) if k % 7 == 4 and k % 9 == 5)
    assert result["estimate"] == x
    assert result["estimate"] % 7 == 4
    assert result["estimate"] % 9 == 5
