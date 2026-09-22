"""Tests for eslprc.esl_perceptron."""

from morie.fn import _array_core as np

from morie.fn.eslprc import esl_perceptron


def test_eslprc_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (100, 5))
    y = rng.choice([-1.0, 1.0], size=100)
    result = esl_perceptron(X, y)
    assert isinstance(result, dict)
    for key in ("estimate", "beta", "n_errors", "epochs",
                "converged", "separable_within_budget",
                "n", "p", "method"):
        assert key in result
    assert result["n"] == 100
    assert result["p"] == 5
    assert len(result["beta"]) == 5
    assert result["estimate"] == result["beta"][0]
    assert result["epochs"] >= 1
    assert result["n_errors"] >= 0
    assert isinstance(result["converged"], bool)
    assert isinstance(result["separable_within_budget"], bool)
    assert result["method"].startswith("Rosenblatt")


def test_eslprc_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (100, 5))
    y = rng.choice([-1.0, 1.0], size=100)
    result = esl_perceptron(X, y)
    assert isinstance(result, dict)
    assert result["n"] == 100
    assert result["p"] == 5
