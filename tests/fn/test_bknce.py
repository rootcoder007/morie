"""Tests for bknce.burkov_noise_contrastive_estimation."""

from morie.fn import _array_core as np

from morie.fn.bknce import burkov_noise_contrastive_estimation


def test_bknce_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    pos_scores = rng.normal(0, 1, 100)
    neg_scores = rng.normal(0, 1, (100, 5))
    noise_prob = rng.uniform(0.1, 1.0, (100, 5))
    result = burkov_noise_contrastive_estimation(pos_scores, neg_scores, noise_prob, 5)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "loss" in result
    assert "pos_loss" in result
    assert "neg_loss" in result
    assert "accuracy" in result
    assert "corrected" in result
    assert result["corrected"] is True
    assert result["k"] == 5
    assert result["n"] == 100
    assert result["cost_ratio"] == 6.0

    # Compute the expected value independently from the formula.
    def logsig(z):
        return -np.logaddexp(0.0, -z)

    Q = np.asarray(noise_prob, dtype=float)
    neg_adj = neg_scores - np.log(5 * Q)
    pos_adj = pos_scores - np.log(5 * np.maximum(Q.mean(axis=1), 1e-300))
    expected_pl = float(-np.mean(logsig(pos_adj)))
    expected_nl = float(-np.mean(np.sum(logsig(-neg_adj), axis=1)))
    expected_loss = expected_pl + expected_nl
    expected_acc = float(np.mean(pos_adj[:, None] > neg_adj))

    assert result["pos_loss"] == expected_pl
    assert result["neg_loss"] == expected_nl
    assert result["loss"] == expected_loss
    assert result["estimate"] == expected_loss
    assert result["accuracy"] == expected_acc
    assert result["objective"] == "noise-contrastive estimation"


def test_bknce_edge():
    """Test edge cases: negative sampling when noise_prob is omitted."""
    rng = np.random.default_rng(42)
    pos_scores = rng.normal(0, 1, 100)
    neg_scores = rng.normal(0, 1, (100, 5))
    # Without noise_prob the function performs plain negative sampling.
    result = burkov_noise_contrastive_estimation(pos_scores, neg_scores, k=5)
    assert isinstance(result, dict)
    assert "loss" in result
    assert "pos_loss" in result
    assert "neg_loss" in result
    assert "accuracy" in result
    assert "corrected" in result
    assert "estimate" in result
    assert "k" in result
    assert "cost_ratio" in result
    assert "n" in result
    assert "objective" in result
    assert result["corrected"] is False
    assert result["k"] == 5
    assert result["n"] == 100
    assert result["cost_ratio"] == 6.0
    assert result["objective"] == "negative sampling"

    # Independent recomputation: negative sampling has no log(k q(w)) correction.
    def logsig(z):
        return -np.logaddexp(0.0, -z)

    expected_pl = float(-np.mean(logsig(pos_scores)))
    expected_nl = float(-np.mean(np.sum(logsig(-neg_scores), axis=1)))
    expected_loss = expected_pl + expected_nl
    expected_acc = float(np.mean(pos_scores[:, None] > neg_scores))

    assert result["pos_loss"] == expected_pl
    assert result["neg_loss"] == expected_nl
    assert result["loss"] == expected_loss
    assert result["estimate"] == expected_loss
    assert result["accuracy"] == expected_acc
