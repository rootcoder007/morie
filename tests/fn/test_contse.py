"""Tests for contse.contrastive_sent."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.contse import contrastive_sent


def test_contse_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n, d = 40, 3
    sentences = rng.normal(0.0, 1.0, (n, d))
    tau = 0.1
    result = contrastive_sent(sentences, tau)
    assert isinstance(result, dict)
    for key in ("estimate", "loss", "per_item", "alignment",
                "uniformity", "n", "d"):
        assert key in result
    assert result["n"] == n
    assert result["d"] == d
    assert math.isfinite(result["estimate"])
    assert math.isfinite(result["loss"])
    assert math.isfinite(result["alignment"])
    assert len(result["per_item"]) == n


def test_contse_edge():
    """Test edge cases."""
    rng = np.random.default_rng(7)
    n, d = 20, 2
    sentences = rng.normal(0.0, 5.0, (n, d))
    result = contrastive_sent(sentences, tau=0.05, dropout=0.0, seed=0)
    assert isinstance(result, dict)
    assert result["n"] == n
    assert result["d"] == d
    assert math.isfinite(result["loss"])
    assert math.isfinite(result["estimate"])
    # bad tau must raise per docstring
    with pytest.raises(ValueError):
        contrastive_sent(sentences, tau=0.0)
    # bad dropout must raise per docstring
    with pytest.raises(ValueError):
        contrastive_sent(sentences, dropout=1.0)
    # empty input must raise per docstring
    with pytest.raises(ValueError):
        contrastive_sent([], tau=0.1)
