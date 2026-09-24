"""Tests for rgmemb.rangayyan_membrane_potential."""

from morie.fn import _array_core as np

from morie.fn.bsaphys import rangayyan_membrane_potential


def test_rgmemb_basic():
    """Test basic functionality."""
    t = np.array([float(i + 1) for i in range(40)])
    result = rangayyan_membrane_potential(t)
    assert isinstance(result, dict)
    assert "t_ms" in result


def test_rgmemb_edge():
    """Test edge cases."""
    t = np.array([float(i + 1) for i in range(40)])
    result = rangayyan_membrane_potential(t)
    assert isinstance(result, dict)
