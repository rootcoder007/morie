"""Tests for km150.kamath_ch9_flamingo_dataset_mix."""

import math

from morie.fn import _array_core as np

from morie.fn.km150 import kamath_ch9_flamingo_dataset_mix


def _make_seq(rng, seq_len):
    """Create one sequence of per-token conditional probabilities."""
    return [float(rng.uniform(0.1, 0.9)) for _ in range(seq_len)]


def _make_dataset(rng, seq_len, n_seqs):
    """Create one dataset: a list of sequences."""
    return [_make_seq(rng, seq_len) for _ in range(n_seqs)]


def test_km150_basic():
    """Test basic functionality with multiple datasets."""
    rng = np.random.default_rng(42)
    D_m = [
        _make_dataset(rng, seq_len=4, n_seqs=3),
        _make_dataset(rng, seq_len=4, n_seqs=2),
    ]
    lambda_m = [0.3, 0.7]

    result = kamath_ch9_flamingo_dataset_mix(D_m, lambda_m)

    assert isinstance(result, dict)
    assert "estimate" in result
    assert math.isfinite(result["estimate"])
    assert "per_dataset_nll" in result
    assert len(result["per_dataset_nll"]) == 2
    assert all(math.isfinite(v) for v in result["per_dataset_nll"])
    assert result["weights"] == [0.3, 0.7]
    assert result["n"] == 2
    assert "method" in result


def test_km150_edge():
    """Test edge case with a single dataset and matching weight."""
    rng = np.random.default_rng(7)
    D_m = [_make_dataset(rng, seq_len=2, n_seqs=1)]
    lambda_m = [1.0]

    result = kamath_ch9_flamingo_dataset_mix(D_m, lambda_m)

    assert isinstance(result, dict)
    assert "estimate" in result
    assert math.isfinite(result["estimate"])
    assert result["n"] == 1
    assert result["weights"] == [1.0]
    assert len(result["per_dataset_nll"]) == 1
    assert "method" in result
