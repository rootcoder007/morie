"""Tests for eslcrm.esl_cross_entropy."""

from morie.fn import _array_core as np

from morie.fn.eslcrm import esl_cross_entropy


def test_eslcrm_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(43)
    n = 100
    K = 5
    # Build random class labels in [0, K-1] and matching probability rows
    # that sum to 1 within tolerance.
    idx = rng.integers(0, K, size=n)
    raw = rng.uniform(0.01, 1.0, size=(n, K))
    raw = raw / raw.sum(axis=1, keepdims=True)

    # One-hot encoding of idx, computed independently from the function.
    Y_one_hot = np.zeros((n, K), dtype=float)
    Y_one_hot[np.arange(n), idx] = 1.0

    # Independent per-observation loss via plain arithmetic on (Y, p).
    with np.errstate(divide="ignore", invalid="ignore"):
        per_obs = (-Y_one_hot * np.log(raw)).sum(axis=1)
    expected_estimate = float(per_obs.mean())

    result = esl_cross_entropy(idx, raw)

    assert isinstance(result, dict)
    assert "estimate" in result
    assert "per_observation" in result
    assert "n" in result
    assert "K" in result
    assert "label_form" in result
    assert "method" in result

    assert result["n"] == n
    assert result["K"] == K
    assert result["label_form"] == "class-index"

    assert abs(result["estimate"] - expected_estimate) < 1e-12
    assert len(result["per_observation"]) == n
    # Per-observation losses are non-negative (since -log(p_k) >= 0 for p_k in (0, 1]).
    assert all(v >= 0.0 for v in result["per_observation"])


def test_eslcrm_edge():
    """Test edge cases: one-hot input and zero predicted probability on true class."""
    # One-hot rows: label_form must be reported as "one-hot".
    y_oh = [[0.0, 1.0]]
    p_ok = [[0.25, 0.75]]
    result = esl_cross_entropy(y_oh, p_ok)
    assert isinstance(result, dict)
    assert result["label_form"] == "one-hot"
    # Independent arithmetic: -1 * log(0.75)
    import math
    expected = -math.log(0.75)
    assert abs(result["estimate"] - expected) < 1e-12
    assert abs(result["per_observation"][0] - expected) < 1e-12

    # Predicted probability of zero on the true class -> infinite loss,
    # returned as inf (NOT silently clipped to a finite number).
    result_inf = esl_cross_entropy([0], [[0.0, 1.0]])
    assert result_inf["estimate"] == float("inf")
    assert result_inf["per_observation"][0] == float("inf")
