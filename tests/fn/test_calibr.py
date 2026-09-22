"""Tests for calibr.calibration_estimator."""

from morie.fn import _array_core as np

from morie.fn.calibr import calibration_estimator


def test_calibr_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    weights = np.random.default_rng(45).exponential(1, 100)
    # totals must have one entry per calibration column (p = X.shape[1])
    totals = X.sum(axis=0) + np.random.default_rng(7).normal(0, 0.01, 5)
    result = calibration_estimator(y, X, weights, totals)
    assert isinstance(result, dict)
    # The function returns a RichResult with documented keys
    for key in (
        "total",
        "calibrated_weights",
        "margins_reproduced",
        "max_margin_error",
        "n_negative",
        "weight_ratio_range",
        "distance",
        "equals_greg",
        "n",
        "p",
        "method",
    ):
        assert key in result

    # chi-square calibration reproduces margins exactly (closed-form)
    assert result["margins_reproduced"] is True
    assert result["max_margin_error"] < 1e-6
    assert result["equals_greg"] is True

    # calibrated_weights length matches n; their weighted X sums equal totals
    w = np.asarray(result["calibrated_weights"], dtype=float)
    n = y.size
    assert w.shape == (n,)
    assert result["n"] == n
    assert result["p"] == X.shape[1]

    achieved = (w[:, None] * X).sum(axis=0)
    expected_totals = np.asarray(totals, dtype=float).ravel()
    assert np.allclose(achieved, expected_totals, rtol=1e-8, atol=1e-8)

    # total must equal sum_i w_i * y_i (independent expression, not copied)
    expected_total = float(np.sum(w * np.asarray(y, dtype=float).ravel()))
    assert abs(result["total"] - expected_total) < 1e-8


def test_calibr_edge():
    """Test edge cases: shape handling and totals arity."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    weights = np.random.default_rng(45).exponential(1, 100)
    # totals length must equal p (= X.shape[1] = 5), not n
    totals = X.sum(axis=0)
    result = calibration_estimator(y, X, weights, totals)
    assert isinstance(result, dict)
    assert result["n"] == 100
    assert result["p"] == 5
