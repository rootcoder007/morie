"""Tests for sasRec.sasrec."""

from morie.fn import _array_core as np

from morie.fn.sasRec import sasrec


def test_sasRec_basic():
    """Test basic functionality."""
    E = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    WQ = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    WK = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    WV = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = sasrec(E, WQ, WK, WV)
    assert isinstance(result, dict)
    assert "output" in result


def test_sasRec_edge():
    """Test edge cases."""
    E = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    WQ = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    WK = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    WV = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = sasrec(E, WQ, WK, WV)
    assert isinstance(result, dict)
