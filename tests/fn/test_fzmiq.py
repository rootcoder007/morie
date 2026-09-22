"""Tests for fzmiq.fauzi_moment_ineq_ustat."""

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

from morie.fn.fzmiq import fauzi_moment_ineq_ustat


def test_fzmiq_basic():
    """Test basic functionality against the documented Eq. (3.12) formula."""
    n = 100
    k = 2
    q = 2.0
    rhomom = 1.0
    c = 1.0
    result = fauzi_moment_ineq_ustat(n, k, q, rhomom=rhomom, c=c)
    assert isinstance(result, dict)
    for key in ("bound_over_c", "bound", "exponent", "naive", "method"):
        assert key in result
    expected_expo = q * k / 2.0
    expected_naive = q * k
    expected_bound_over_c = (float(n) ** expected_expo) * rhomom
    expected_bound = c * expected_bound_over_c
    assert result["exponent"] == expected_expo
    assert result["naive"] == expected_naive
    assert result["bound_over_c"] == expected_bound_over_c
    assert result["bound"] == expected_bound
    assert result["method"] == "H-decomposition moment bound (Eq. 3.12)"


def test_fzmiq_edge():
    """Test edge cases for the documented constraints."""
    n = 1
    k = 1
    q = 2.0
    result = fauzi_moment_ineq_ustat(n, k, q)
    assert isinstance(result, dict)
    assert result["exponent"] == q * k / 2.0
    assert result["naive"] == q * k
    assert result["bound_over_c"] == float(n) ** (q * k / 2.0)
    assert result["bound"] == 1.0 * (float(n) ** (q * k / 2.0))

    import pytest
    with pytest.raises(ValueError):
        fauzi_moment_ineq_ustat(0, 1, 2.0)
    with pytest.raises(ValueError):
        fauzi_moment_ineq_ustat(10, 0, 2.0)
    with pytest.raises(ValueError):
        fauzi_moment_ineq_ustat(10, 1, 1.0)
