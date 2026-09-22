"""Tests for ghs029.ghosal_ch3_polya_tree_density_moments."""

from morie.fn import _array_core as np

from morie.fn.ghs029 import ghosal_ch3_polya_tree_density_moments


def test_ghs029_basic():
    """Test basic functionality against the documented formula (eq. 3.22)."""
    rng = np.random.default_rng(42)
    alpha_path = [(0.05, 0.05), (0.10, 0.20), (0.30, 0.40)]
    result = ghosal_ch3_polya_tree_density_moments(alpha_path)

    assert isinstance(result, dict)
    assert "estimate" in result
    assert "value" in result
    assert "second_moment" in result

    # Independent computation of the documented formula.
    m1 = 1.0
    m2 = 1.0
    for a_take, a_other in [(float(a), float(b)) for a, b in alpha_path]:
        s = a_take + a_other
        m1 *= 2.0 * a_take / s
        m2 *= 4.0 * a_take * (a_take + 1.0) / (s * (s + 1.0))

    assert result["estimate"] == m1
    assert result["second_moment"] == m2
    assert result["value"] == [m1, m2]


def test_ghs029_depth_truncation():
    """depth argument should truncate the path and recompute accordingly."""
    rng = np.random.default_rng(42)
    alpha_path = [(0.05, 0.05), (0.10, 0.20), (0.30, 0.40)]

    full = ghosal_ch3_polya_tree_density_moments(alpha_path)
    truncated = ghosal_ch3_polya_tree_density_moments(alpha_path, depth=2)

    # Independent recomputation for depth=2.
    m1 = 1.0
    m2 = 1.0
    for a_take, a_other in [(float(a), float(b)) for a, b in alpha_path[:2]]:
        s = a_take + a_other
        m1 *= 2.0 * a_take / s
        m2 *= 4.0 * a_take * (a_take + 1.0) / (s * (s + 1.0))

    assert truncated["estimate"] == m1
    assert truncated["second_moment"] == m2
    assert truncated["value"] == [m1, m2]
    assert truncated["estimate"] != full["estimate"]


def test_ghs029_edge():
    """Edge case: a single-level tree."""
    alpha_path = [(0.5, 0.5)]
    result = ghosal_ch3_polya_tree_density_moments(alpha_path)

    a_take, a_other = 0.5, 0.5
    s = a_take + a_other
    m1 = 2.0 * a_take / s
    m2 = 4.0 * a_take * (a_take + 1.0) / (s * (s + 1.0))

    assert isinstance(result, dict)
    assert result["estimate"] == m1
    assert result["second_moment"] == m2
    assert result["value"] == [m1, m2]
