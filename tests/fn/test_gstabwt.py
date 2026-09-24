"""Tests for gstabwt.stabilized_weights."""

from morie.fn import _array_core as np

from morie.fn.gstabwt import stabilized_weights


def test_gstabwt_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 40
    nt = 3
    # numerator and denominator models: n subjects x nt time points,
    # values in (0, 1) so denominator probabilities stay positive.
    denominator_model = rng.uniform(0.1, 0.9, (n, nt))
    numerator_model = rng.uniform(0.1, 0.9, (n, nt))
    treatment = rng.integers(0, 2, n)
    history = rng.integers(0, 2, n)
    result = stabilized_weights(treatment, history, numerator_model, denominator_model)
    assert isinstance(result, dict)
    assert "weights" in result
    assert "unstabilized" in result
    assert "mean_weight" in result
    assert "max_weight" in result
    assert "n" in result
    assert "n_times" in result
    assert "method" in result
    assert result["n"] == n
    assert result["n_times"] == nt
    assert len(result["weights"]) == n
    assert len(result["unstabilized"]) == n


def test_gstabwt_edge():
    """Test edge cases - numerator_model defaults to None (uniform marginal)."""
    import math
    rng = np.random.default_rng(42)
    n = 5
    nt = 2
    denominator_model = rng.uniform(0.1, 0.9, (n, nt))
    # Omit numerator_model; should fall back to uniform marginal of 1.0.
    result = stabilized_weights(denominator_model=denominator_model)
    assert isinstance(result, dict)
    assert "weights" in result
    assert "mean_weight" in result
    assert result["n"] == n
    assert result["n_times"] == nt
    assert len(result["weights"]) == n
    assert math.isfinite(result["mean_weight"])
    assert result["max_weight"] >= result["mean_weight"]
