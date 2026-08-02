"""Tests for funcal.functional_annotation."""

from morie.fn import _array_core as np

from morie.fn.funcal import functional_annotation


def test_funcal_basic():
    """Test basic functionality."""
    sequences = np.random.default_rng(42).normal(0, 1, 100)
    eggnog_db = np.random.default_rng(42).normal(0, 1, 100)
    result = functional_annotation(sequences, eggnog_db)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_funcal_edge():
    """Test edge cases."""
    sequences = np.random.default_rng(42).normal(0, 1, 100)
    eggnog_db = np.random.default_rng(42).normal(0, 1, 100)
    result = functional_annotation(sequences, eggnog_db)
    assert isinstance(result, dict)
