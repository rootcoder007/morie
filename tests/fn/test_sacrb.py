"""Tests for sacrb.sacrebleu."""

from morie.fn import _array_core as np

from morie.fn.sacrb import sacrebleu


def test_sacrb_basic():
    """Test basic functionality."""
    candidates = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    references = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = sacrebleu(candidates, references)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_sacrb_edge():
    """Test edge cases."""
    candidates = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    references = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = sacrebleu(candidates, references)
    assert isinstance(result, dict)
