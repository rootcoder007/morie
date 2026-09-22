"""Tests for esllgr.esl_logistic_reg."""

from morie.fn import _array_core as np

from morie.fn.esllgr import esl_logistic_reg


def test_esllgr_basic():
    """Test basic functionality."""
    rng_x = np.random.default_rng(42)
    X = rng_x.normal(0, 1, (100, 5))
    rng_y = np.random.default_rng(43)
    eta = -0.5 + 0.8 * X[:, 0] - 0.6 * X[:, 1] + 0.4 * X[:, 2]
    p = 1.0 / (1.0 + np.exp(-eta))
    y = (rng_y.random(100) < p).astype(float)
    result = esl_logistic_reg(X, y)

    # Check that result is a mapping-like object with the documented keys
    assert hasattr(result, "__getitem__")
    assert "beta" in result
    assert "se" in result
    assert "p_value" in result
    assert "prob" in result
    assert "class_" in result
    assert "odds_ratio" in result
    assert "loglik" in result
    assert "deviance" in result
    assert "accuracy" in result
    assert "confusion" in result

    # Probabilities in [0, 1]
    assert result["prob"].min() >= 0.0
    assert result["prob"].max() <= 1.0

    # Class rule is binary
    unique_classes = np.unique(result["class_"])
    assert set(unique_classes.tolist()).issubset({0, 1})

    # Odds ratio equals exp(beta) (documented relationship)
    beta = np.asarray(result["beta"])
    odds = np.asarray(result["odds_ratio"])
    assert np.allclose(odds, np.exp(beta))

    # Accuracy is between 0 and 1
    acc = result["accuracy"]
    assert 0.0 <= acc <= 1.0

    # Confusion matrix is 2x2
    conf = np.asarray(result["confusion"])
    assert conf.shape == (2, 2)

    # The classifier beats chance on this separable-enough signal
    assert acc > 0.5


def test_esllgr_edge():
    """Test edge cases."""
    rng_x = np.random.default_rng(42)
    X = rng_x.normal(0, 1, (100, 5))
    rng_y = np.random.default_rng(43)
    eta = -0.3 + 1.0 * X[:, 0]
    p = 1.0 / (1.0 + np.exp(-eta))
    y = (rng_y.random(100) < p).astype(float)
    result = esl_logistic_reg(X, y, threshold=0.3)

    # Threshold must be in (0, 1); an invalid value raises ValueError
    import pytest
    with pytest.raises(ValueError):
        esl_logistic_reg(X, y, threshold=1.5)

    # With a lower threshold, more observations are classified as 1
    cls_lower = np.asarray(result["class_"])
    rng_x2 = np.random.default_rng(42)
    X2 = rng_x2.normal(0, 1, (100, 5))
    rng_y2 = np.random.default_rng(43)
    eta2 = -0.3 + 1.0 * X2[:, 0]
    p2 = 1.0 / (1.0 + np.exp(-eta2))
    y2 = (rng_y2.random(100) < p2).astype(float)
    result_default = esl_logistic_reg(X2, y2, threshold=0.5)
    cls_default = np.asarray(result_default["class_"])
    assert cls_lower.sum() >= cls_default.sum()
