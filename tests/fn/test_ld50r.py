"""Tests for ld50r.acute_toxicity_ld50."""

from morie.fn import _array_core as np

from morie.fn.ld50r import acute_toxicity_ld50


def test_ld50r_basic():
    """Test basic functionality."""
    dose = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    n_dead = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    n_total = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = acute_toxicity_ld50(dose, n_dead, n_total)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_ld50r_edge():
    """Test edge cases."""
    dose = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    n_dead = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    n_total = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = acute_toxicity_ld50(dose, n_dead, n_total)
    assert isinstance(result, dict)
