"""Tests for eslada.esl_adaboost."""

from morie.fn import _array_core as np

from morie.fn.eslada import esl_adaboost


def test_eslada_basic():
    """Test basic functionality on linearly separable data."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (40, 3))
    # Labels in {-1, +1} determined by sign of first feature, per docstring.
    y = np.where(X[:, 0] >= 0, 1.0, -1.0)
    M = 50
    result = esl_adaboost(X, y, M)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "statistic" not in result  # documented key is "estimate", not "statistic"
    assert "stumps" in result
    assert "alphas" in result
    assert "rounds_used" in result
    assert "prediction" in result
    assert "margin" in result
    assert "n" in result and "p" in result
    assert "method" in result
    # result["estimate"] is the training error rate from the documented formula:
    # mean of (committee != y), where committee = sign(sum_m alpha_m * stump_m(x)).
    n = X.shape[0]
    # Independent expression of training error: 1 if any prediction mismatches y.
    expected_estimate = float(np.mean(np.asarray(result["prediction"]) != y))
    assert result["estimate"] == expected_estimate
    assert result["estimate"] >= 0.0 and result["estimate"] <= 1.0
    assert result["n"] == n
    assert result["p"] == 3
    assert len(result["alphas"]) == result["rounds_used"]
    assert len(result["stumps"]) == result["rounds_used"]
    assert result["rounds_used"] <= M


def test_eslada_edge():
    """Test edge case: first stump perfectly separates the threshold problem."""
    # From the docstring's worked example: a threshold problem solved by the
    # first stump. Labels in {-1, +1}, shape (n,) for y and (n, p) for X.
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([1.0, 1.0, -1.0, -1.0])
    result = esl_adaboost(X, y)
    assert isinstance(result, dict)
    # First stump is perfect, so it should stop after one round.
    assert result["rounds_used"] == 1
    # With a perfect stump, training error is exactly 0 by the documented formula.
    assert result["estimate"] == 0.0
    # Independent expression of training error computed from the prediction list.
    expected_estimate = float(np.mean(np.asarray(result["prediction"]) != y))
    assert result["estimate"] == expected_estimate
    # Alpha for a perfect stump is the documented sentinel value 10.0 (infinite
    # log((1-err)/err) clipped to 10.0).
    assert result["alphas"] == [10.0]
    # Predictions on the training set must match the labels exactly.
    assert all(int(p) == int(lbl) for p, lbl in zip(result["prediction"], y))
