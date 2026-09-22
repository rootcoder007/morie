"""Tests for gpregb.gp_regression_bayes."""

from morie.fn import _array_core as np

from morie.fn.gpregb import gp_regression_bayes


def _grid_inputs(seed=0, n=20, d=2):
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, (n, d)).tolist()
    y = rng.normal(0.0, 1.0, n).tolist()
    X_test = rng.normal(0.0, 1.0, (5, d)).tolist()
    lengthscales = [0.5, 1.0, 2.0]
    noises = [0.01, 0.1]
    return X, y, X_test, lengthscales, noises


def test_gpregb_basic():
    """Test basic functionality on a small grid-friendly dataset."""
    X, y, X_test, lengthscales, noises = _grid_inputs(seed=42, n=20, d=3)

    result = gp_regression_bayes(
        X, y, kernel=None, X_test=X_test,
        lengthscales=lengthscales, noises=noises, variance=1.0,
    )

    assert isinstance(result, dict)
    # 'mean' and 'variance' are the documented predictive outputs
    assert "mean" in result
    assert "variance" in result
    assert "estimate" in result
    assert "weights" in result
    assert "loglik" in result
    assert "n" in result
    assert "method" in result

    # Shapes match the number of test points
    m = len(X_test)
    assert len(result["mean"]) == m
    assert len(result["variance"]) == m
    assert len(result["weights"]) == len(lengthscales) * len(noises)
    assert len(result["loglik"]) == len(lengthscales) * len(noises)

    # Weights form a probability vector
    w = result["weights"]
    assert abs(sum(w) - 1.0) < 1e-12
    for wi in w:
        assert wi >= 0.0

    # Predictive variance is non-negative
    for v in result["variance"]:
        assert v >= -1e-12


def test_gpregb_edge():
    """Test edge cases: defaults and a length-1 grid (estimate only)."""
    X, y, _, _, _ = _grid_inputs(seed=7, n=15, d=2)

    # Use documented defaults for lengthscales/noises/X_test; only pass X, y.
    result = gp_regression_bayes(X, y)

    assert isinstance(result, dict)
    assert "mean" in result
    assert "variance" in result
    assert "weights" in result
    assert "loglik" in result

    # Defaults: grid = 3 lengthscales * 2 noises = 6
    assert len(result["weights"]) == 6
    assert len(result["loglik"]) == 6
    assert abs(sum(result["weights"]) - 1.0) < 1e-12

    # estimate == mean[0] for the single-X_test case (X_test defaults to X)
    assert result["estimate"] == result["mean"][0]

    # Determinism: same inputs -> same outputs
    result2 = gp_regression_bayes(X, y)
    assert result["mean"] == result2["mean"]
    assert result["variance"] == result2["variance"]
