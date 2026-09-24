"""Tests for hmmlb.geron_multilabel."""

import math

from morie.fn import _array_core as np

from morie.fn.hmmlb import geron_multilabel


def test_hmmlb_basic():
    """Test basic functionality with supplied predictions."""
    rng = np.random.default_rng(42)
    m, n_features, K = 40, 3, 4
    X = rng.normal(0, 1, (m, n_features))
    Y = rng.integers(0, 2, (m, K))
    Y_pred = rng.integers(0, 2, (m, K))
    result = geron_multilabel(X, Y, Y_pred=Y_pred)
    assert isinstance(result, dict)
    for key in (
        "Y_pred",
        "subset_accuracy",
        "hamming_loss",
        "jaccard",
        "per_label_f1",
        "macro_f1",
        "zero_baseline_hamming",
        "estimate",
        "n",
        "method",
    ):
        assert key in result
    assert result["n"] == m
    assert math.isfinite(result["hamming_loss"])
    assert 0.0 <= result["hamming_loss"] <= 1.0
    assert math.isfinite(result["subset_accuracy"])
    assert 0.0 <= result["subset_accuracy"] <= 1.0
    assert math.isfinite(result["jaccard"])
    assert 0.0 <= result["jaccard"] <= 1.0
    assert math.isfinite(result["zero_baseline_hamming"])
    assert 0.0 <= result["zero_baseline_hamming"] <= 1.0
    assert math.isfinite(result["macro_f1"])


def test_hmmlb_edge():
    """Test edge case with small, sparse-label input and supplied predictions."""
    rng = np.random.default_rng(42)
    m, n_features, K = 6, 2, 3
    X = rng.normal(0, 1, (m, n_features))
    Y = rng.integers(0, 2, (m, K))
    Y_pred = rng.integers(0, 2, (m, K))
    result = geron_multilabel(X, Y, k=2, Y_pred=Y_pred)
    assert isinstance(result, dict)
    assert result["n"] == m
    assert math.isfinite(result["hamming_loss"])
    assert 0.0 <= result["hamming_loss"] <= 1.0
    assert math.isfinite(result["subset_accuracy"])
    assert 0.0 <= result["subset_accuracy"] <= 1.0
    assert math.isfinite(result["jaccard"])
    assert 0.0 <= result["jaccard"] <= 1.0
