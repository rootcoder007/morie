"""Tests for finalsz.final_epidemic_size."""

from math import exp

from morie.fn import _array_core as np

from morie.fn.finalsz import final_epidemic_size


def _solve_final_size(R0, s0=1.0, i0=None, tol=1e-14, max_iter=200):
    """Independent bisection of Z = s0 (1 - exp(-R0 (Z + i0)))."""
    if i0 is None:
        i0 = 1.0 - s0
    def resid(Z):
        return s0 * (1.0 - exp(-R0 * (Z + i0))) - Z
    lo, hi = 0.0, s0
    if i0 == 0.0 and R0 * s0 <= 1.0:
        return 0.0
    for _ in range(int(max_iter)):
        mid = 0.5 * (lo + hi)
        if resid(mid) > 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def test_finalsz_basic():
    """Test basic functionality: classical limit s0=1, i0=0, well above threshold."""
    R0 = 2.5
    s0 = 1.0
    result = final_epidemic_size(R0, s0)

    expected_Z = _solve_final_size(R0, s0=s0, i0=0.0)

    assert isinstance(result, dict)
    assert "estimate" in result
    assert "final_size" in result
    assert "s_inf" in result
    assert "attack_rate" in result
    assert "R0" in result
    assert "s0" in result
    assert "i0" in result
    assert "residual" in result
    assert "iters" in result
    assert "n" in result
    assert "method" in result

    assert abs(result["estimate"] - expected_Z) < 1e-10
    assert abs(result["final_size"] - expected_Z) < 1e-10
    assert abs(result["s_inf"] - (s0 - expected_Z)) < 1e-12
    assert abs(result["attack_rate"] - expected_Z / s0) < 1e-12
    assert result["R0"] == R0
    assert result["s0"] == s0
    assert result["i0"] == 0.0
    assert abs(result["residual"]) < 1e-9
    # Classical identity Z = 1 - exp(-R0 * Z)
    assert abs(result["final_size"] - (1.0 - exp(-R0 * expected_Z))) < 1e-10


def test_finalsz_edge():
    """Test edge case: R0*s0 <= 1 with i0=0 -> no epidemic, Z = 0."""
    R0 = 0.7
    s0 = 1.0
    result = final_epidemic_size(R0, s0)

    assert isinstance(result, dict)
    assert result["final_size"] == 0.0
    assert result["estimate"] == 0.0
    assert result["s_inf"] == s0
    assert result["attack_rate"] == 0.0
    assert result["residual"] == 0.0
    assert result["R0"] == R0
    assert result["s0"] == s0
    assert result["i0"] == 0.0
