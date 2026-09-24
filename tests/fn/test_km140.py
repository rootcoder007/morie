"""Tests for km140.kamath_ch9_moc_loss."""
import math

from morie.fn import _array_core as np

from morie.fn.km140 import kamath_ch9_moc_loss


def test_km140_basic():
    """Test basic functionality with integer ground-truth labels."""
    rng = np.random.default_rng(42)
    M, T = 5, 3
    # Build an M x T softmax distribution over object classes (each row sums to 1).
    raw = rng.uniform(0.0, 1.0, (M, T))
    g_theta = [[float(x) / sum(row) for x in row] for row in raw]
    # Integer ground-truth class indices, one per masked region.
    labels = [int(c) for c in rng.integers(0, T, M)]
    v = [[0.0]]  # unused when g_theta is supplied as an array.
    result = kamath_ch9_moc_loss(None, None, v, g_theta, labels=labels)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "as_printed" in result
    assert "per_region" in result
    assert "true_class_probability" in result
    assert "n_masked_regions" in result
    assert "n" in result
    assert "method" in result
    assert result["n_masked_regions"] == M
    assert result["n"] == M
    assert math.isfinite(result["estimate"])
    assert result["as_printed"] == -result["estimate"]
    assert len(result["per_region"]) == M
    assert len(result["true_class_probability"]) == M
    assert all(p >= 0 for p in result["true_class_probability"])
    assert all(c >= 0 for c in result["per_region"])


def test_km140_edge():
    """Test one-hot ground-truth labels."""
    rng = np.random.default_rng(43)
    M, T = 4, 3
    raw = rng.uniform(0.0, 1.0, (M, T))
    g_theta = [[float(x) / sum(row) for x in row] for row in raw]
    labels_idx = [int(c) for c in rng.integers(0, T, M)]
    onehot = [[0.0] * T for _ in range(M)]
    for i, c in enumerate(labels_idx):
        onehot[i][c] = 1.0
    v = [[0.0]]
    result = kamath_ch9_moc_loss(None, None, v, g_theta, labels=onehot)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert math.isfinite(result["estimate"])
    assert result["n_masked_regions"] == M
    assert result["n"] == M
    assert result["as_printed"] == -result["estimate"]
    assert result["method"] == "masked object classification loss (Kamath Eq 9.12)"
    assert len(result["per_region"]) == M
    assert len(result["true_class_probability"]) == M
