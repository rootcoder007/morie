"""Tests for fzc1x.fauzi_c1_coefficient."""

from morie.fn import _array_core as np

from morie.fn.fzc1x import fauzi_c1_coefficient


def test_fzc1x_basic():
    """Test basic functionality."""
    dg = 1.0  # g'(g^{-1}(x)) -- identity transform gives g' = 1
    d2g = 0.0  # g''(g^{-1}(x)) -- identity transform gives g'' = 0
    density = 0.4  # f_X(x)
    fp = -0.2  # f_X'(x)

    result = fauzi_c1_coefficient(dg, d2g, density, fp)

    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result
    assert result["method"] == "c_1 bias coefficient of the boundary-free KDFE (Eq. 5.8)"

    # Independent computation from Eq. (5.8):
    #   c_1(x) = g''(g^{-1}(x)) * f_X(x) + [g'(g^{-1}(x))]^2 * f_X'(x)
    expected = d2g * density + (dg ** 2) * fp
    assert abs(result["estimate"] - expected) < 1e-12


def test_fzc1x_edge():
    """Test edge cases: identity transformation matches the naive KDFE bias."""
    # For the identity transformation g(v) = v:
    #   g'(g^{-1}(x)) = 1, g''(g^{-1}(x)) = 0
    # so c_1(x) reduces to f_X'(x) (the naive KDFE bias coefficient, before
    # the 1/2 * mu_2(K) factor).
    dg = 1.0
    d2g = 0.0
    density = 0.4
    fp = -0.2

    result = fauzi_c1_coefficient(dg, d2g, density, fp)

    assert isinstance(result, dict)
    assert "estimate" in result
    assert result["method"] == "c_1 bias coefficient of the boundary-free KDFE (Eq. 5.8)"

    expected = fp  # 0 * 0.4 + 1^2 * (-0.2) = -0.2
    assert abs(result["estimate"] - expected) < 1e-12

    # All arguments should be coerced to floats regardless of numeric type.
    int_result = fauzi_c1_coefficient(1, 0, 1, 2)
    assert abs(int_result["estimate"] - 2.0) < 1e-12
