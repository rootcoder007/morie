"""Tests for btoob.boot_oob_error."""

from morie.fn import _array_core as np

from morie.fn.btoob import boot_oob_error


def test_btoob_basic():
    """Test basic functionality."""
    rng_x = np.random.default_rng(42)
    rng_y = np.random.default_rng(43)
    x = rng_x.normal(0, 1, (100, 2))
    y = rng_y.normal(0, 1, 100)

    def fit_fn(xb, yb):
        # Simple mean predictor fit on the bootstrap sample.
        return float(np.mean(yb))

    def predict_fn(fitted, x_new):
        # Predict the learned mean for every row of x_new.
        return np.full(x_new.shape[0], fitted)

    result = boot_oob_error(x, y, fit_fn, predict_fn, B=100, seed=0)

    # The documented return is a RichResult; assert the documented keys.
    assert "err_oob" in result
    assert "err_apparent" in result
    assert "per_observation" in result
    assert "n_dropped" in result
    assert "oob_fraction" in result
    assert "B" in result
    assert "n" in result
    assert "method" in result

    # Honest OOB error must not be smaller than the apparent (training) error.
    assert result["err_oob"] >= result["err_apparent"]

    # Recompute the apparent error independently from the formula and compare.
    fitted_full = fit_fn(x, y)
    pred_full = predict_fn(fitted_full, x)
    expected_err_app = float(np.mean((y - pred_full) ** 2))
    assert abs(result["err_apparent"] - expected_err_app) < 1e-12

    # oob_fraction is the share of (i, b) pairs that are out-of-bag; the
    # theory value is (1 - 1/n)^n ~ e^{-1} ~ 0.3679 for the per-replicate
    # inclusion probability, and the same number is the average OOB rate.
    assert 0.25 < result["oob_fraction"] < 0.45
    assert result["B"] == 100
    assert result["n"] == 100


def test_btoob_edge():
    """Test edge cases: small B drops observations that are never OOB."""
    rng_x = np.random.default_rng(42)
    rng_y = np.random.default_rng(43)
    x = rng_x.normal(0, 1, (20, 2))
    y = rng_y.normal(0, 1, 20)

    def fit_fn(xb, yb):
        return float(np.mean(yb))

    def predict_fn(fitted, x_new):
        return np.full(x_new.shape[0], fitted)

    # B small enough that some points are likely never OOB, exercising
    # the n_dropped accounting.
    result = boot_oob_error(x, y, fit_fn, predict_fn, B=5, seed=0)

    assert "n_dropped" in result
    assert result["n_dropped"] >= 0
    assert result["n_dropped"] + int(np.sum(
        np.isfinite(np.asarray(result["per_observation"])))) == 20
