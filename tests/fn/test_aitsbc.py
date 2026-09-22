"""Tests for aitsbc.aitchison_subcomposition."""

from morie.fn import _array_core as np

from morie.fn.aitsbc import aitchison_subcomposition


def test_aitsbc_basic():
    """Test basic functionality."""
    x = np.array([0.5, 0.2, 0.2, 0.1])
    idx = [1, 3]
    result = aitchison_subcomposition(x, idx)
    assert isinstance(result, dict)
    assert "composition" in result
    assert "parts" in result
    assert "total" in result
    assert "D_sub" in result
    assert "D" in result

    expected_sub = [x[i - 1] for i in idx]
    s = sum(expected_sub)
    expected_composition = [1.0 * v / s for v in expected_sub]
    assert list(result["composition"]) == expected_composition
    assert list(result["parts"]) == idx
    assert result["total"] == 1.0
    assert result["D_sub"] == 2
    assert result["D"] == 4
    assert abs(sum(result["composition"]) - 1.0) < 1e-12


def test_aitsbc_edge():
    """Test edge cases."""
    x = np.array([0.1, 0.2, 0.3, 0.4])
    idx = [2, 4]
    result = aitchison_subcomposition(x, idx)
    assert isinstance(result, dict)
    assert "composition" in result
    assert "D_sub" in result

    expected_sub = [x[i - 1] for i in idx]
    s = sum(expected_sub)
    expected_composition = [1.0 * v / s for v in expected_sub]
    assert list(result["composition"]) == expected_composition
    assert result["D_sub"] == 2
    assert abs(sum(result["composition"]) - 1.0) < 1e-12
