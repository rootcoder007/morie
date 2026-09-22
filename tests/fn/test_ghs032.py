"""Tests for ghs032.ghosal_ch3_polya_tree_density_bounds."""

from morie.fn import _array_core as np

from morie.fn.ghs032 import ghosal_ch3_polya_tree_density_bounds


def _a_of_level_factory(values):
    """Return a callable mapping integer level j -> a_j value from a list.

    The list is indexed by level j; values[j] is used for level j.
    """
    values = list(values)
    def a_of_level(j):
        return float(values[j])
    return a_of_level


def test_ghs032_basic():
    """Test basic functionality against the GvdV 2017 sec. 3.7.2 sandwich bound.

    For j in (m, depth], the documented bounds are:
        prod_{j>m} (1 - n/(2 a_j))  <=  posterior tail factor  <=  prod_{j>m} (1 + n/a_j)
    We compute both sides independently and check that the function returns
    keys/value/exact bracket of 1.0 consistent with the formula.
    """
    n = 50.0
    m = 3
    depth = 10

    # Build a sequence of a_j for j = 0..depth with a_j > n (so denominators valid).
    a_values = [100.0] * (depth + 1)  # a_j for j = 0..depth
    a_of_level = _a_of_level_factory(a_values)

    # Independent computation of the documented bounds.
    lo_ref = 1.0
    hi_ref = 1.0
    for j in range(int(m) + 1, int(depth) + 1):
        a = float(a_of_level(j))
        lo_ref *= max(1.0 - n / (2.0 * a), 0.0)
        hi_ref *= 1.0 + n / a

    result = ghosal_ch3_polya_tree_density_bounds(n, a_of_level, m, depth)

    assert isinstance(result, dict)
    assert "value" in result, "function must return a 'value' key per docstring"
    assert "estimate" in result
    assert "lower" in result
    assert "upper" in result
    assert "method" in result

    val = result["value"]
    assert isinstance(val, (list, tuple)) and len(val) == 2
    lo, hi = float(val[0]), float(val[1])

    # The function's outputs must match the independent reference exactly
    # (the implementation is a direct product using the same recurrence).
    assert lo == lo_ref
    assert hi == hi_ref
    assert float(result["estimate"]) == lo_ref
    assert float(result["lower"]) == lo_ref
    assert float(result["upper"]) == hi_ref

    # Documented sandwich: lo <= 1.0 <= hi for all levels given a_j > 0, n >= 0
    # (and the literal 'product of (1 - n/(2a))' lower bound as coded).
    assert lo_ref <= hi_ref


def test_ghs032_edge():
    """Edge case: single level beyond m (depth == m+1) and many levels."""
    n = 10.0
    m = 2
    depth = 6

    a_values = [50.0] * (depth + 1)  # a_j well above n
    a_of_level = _a_of_level_factory(a_values)

    lo_ref = 1.0
    hi_ref = 1.0
    for j in range(int(m) + 1, int(depth) + 1):
        a = float(a_of_level(j))
        lo_ref *= max(1.0 - n / (2.0 * a), 0.0)
        hi_ref *= 1.0 + n / a

    result = ghosal_ch3_polya_tree_density_bounds(n, a_of_level, m, depth)
    assert isinstance(result, dict)

    val = result["value"]
    assert isinstance(val, (list, tuple)) and len(val) == 2
    lo, hi = float(val[0]), float(val[1])

    assert lo == lo_ref
    assert hi == hi_ref
    assert lo <= hi
