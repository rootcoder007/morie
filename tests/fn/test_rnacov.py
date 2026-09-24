"""Tests for rnacov.rna_covariance."""

from morie.fn import _array_core as np
import math
import pytest

from morie.fn.rnacov import rna_covariance


def _make_alignment(rng, n_seq, length):
    """Generate a random RNA alignment as a list of equal-length strings."""
    alphabet = ['A', 'C', 'G', 'U']
    alignment = []
    for _ in range(n_seq):
        indices = rng.integers(0, 4, length)
        seq = ''.join(alphabet[i] for i in indices)
        alignment.append(seq)
    return alignment


def test_rnacov_basic():
    """Test basic functionality with a small alignment and a given structure."""
    rng = np.random.default_rng(42)
    n_seq = 6
    length = 20
    alignment = _make_alignment(rng, n_seq, length)
    structure = "(((...)))(((...))).."  # 9+9+2 = 20 chars, balanced
    result = rna_covariance(alignment, structure)

    # The function returns a dict-like RichResult
    assert isinstance(result, dict)

    # All expected keys from the return payload
    expected_keys = {
        "pair_i", "pair_j", "mutual_information", "support",
        "cells_seen", "n_pairs", "n_weak", "n_covarying",
        "covarying", "total"
    }
    assert expected_keys.issubset(result.keys())

    # Lists should have length equal to n_pairs
    n_pairs = result["n_pairs"]
    assert len(result["pair_i"]) == n_pairs
    assert len(result["pair_j"]) == n_pairs
    assert len(result["mutual_information"]) == n_pairs
    assert len(result["support"]) == n_pairs
    assert len(result["cells_seen"]) == n_pairs

    # Total should be a finite number
    assert isinstance(result["total"], (int, float))
    assert math.isfinite(result["total"])


def test_rnacov_edge():
    """Test that an empty alignment raises a ValueError."""
    structure = "(((...)))(((...))).."
    with pytest.raises(ValueError):
        rna_covariance([], structure)
