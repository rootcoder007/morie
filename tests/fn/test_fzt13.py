"""Tests for fzt13.fauzi_thm1_3_mise_mgkde."""

from morie.fn import _array_core as np

from morie.fn.fzt13 import fauzi_thm1_3_mise_mgkde


def test_fzt13_basic():
    """Test basic functionality against the documented Theorem 1.3 formula."""
    # Theorem 1.3 requires f, fp, fpp, fppp at the evaluation point x >= 0.
    # Use a single evaluation point and a single bandwidth, as documented.
    x = 1.0
    h = 0.3
    f = 0.5
    fp = 0.1
    fpp = -0.2
    fppp = 0.05

    result = fauzi_thm1_3_mise_mgkde(x, h, f, fp, fpp, fppp)

    # The function returns a RichResult; treat it dict-like for the test.
    assert isinstance(result, dict)

    # Documented return keys.
    assert "bias" in result
    assert "a" in result
    assert "b" in result
    assert "h" in result
    assert "book" in result
    assert "method" in result

    # Bandwidth is echoed back.
    assert result["h"] == h
    assert result["book"] is True

    # Independent recomputation of the documented formula (Eqs. 1.15-1.17,
    # book=True form).
    a_indep = fp + 0.5 * x * x * fpp
    b_indep = x + 0.5 * fpp + x * x * (x / 3.0 + 0.5) * fppp
    bias_indep = -2.0 * (b_indep - a_indep * a_indep / (2.0 * f)) * h

    assert abs(result["a"] - a_indep) < 1e-12
    assert abs(result["b"] - b_indep) < 1e-12
    assert abs(result["bias"] - bias_indep) < 1e-12


def test_fzt13_edge():
    """Test edge cases on the documented input constraints."""
    x = 0.5
    h = 0.1
    f = 1.0
    fp = 0.0
    fpp = 0.0
    fppp = 0.0

    result = fauzi_thm1_3_mise_mgkde(x, h, f, fp, fpp, fppp)
    assert isinstance(result, dict)
    assert "bias" in result

    # With all derivatives zero, a = 0 and b = x, so bias = -2 * x * h.
    a_indep = fp + 0.5 * x * x * fpp
    b_indep = x + 0.5 * fpp + x * x * (x / 3.0 + 0.5) * fppp
    bias_indep = -2.0 * (b_indep - a_indep * a_indep / (2.0 * f)) * h
    assert abs(result["bias"] - bias_indep) < 1e-12
