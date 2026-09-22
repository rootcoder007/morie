"""Tests for btbg.boot_bagging_predict."""

from morie.fn import _array_core as np

from morie.fn.btbg import boot_bagging_predict


def test_btbg_basic():
    """Test basic functionality."""
    rng_m = np.random.default_rng(42)
    rng_x = np.random.default_rng(42)
    # models is B-by-m: shape (5, 3) -> 5 replicates, 3 cases.
    models = rng_m.normal(0, 1, (5, 3))
    # X_new is checked only for length -> must have length m = 3.
    X_new = [rng_x.normal(0, 1) for _ in range(3)]
    kind = "regression"
    result = boot_bagging_predict(models, X_new, kind)

    assert isinstance(result, dict)
    assert "y_pred" in result
    assert "vote_share" in result
    assert "B" in result
    assert "m" in result
    assert "estimate" in result

    y_pred = result["y_pred"]
    vote_share = result["vote_share"]

    assert len(y_pred) == 3
    assert len(vote_share) == 3

    # Independent computation of the per-case mean (regression aggregation).
    raw_rows = [[float(v) for v in row] for row in models]
    B = len(raw_rows)
    for j in range(3):
        col = [raw_rows[b][j] for b in range(B)]
        expected_mu = sum(col) / B
        assert abs(y_pred[j] - expected_mu) < 1e-12

        # Independent sample standard deviation (B - 1 in the denominator).
        mean = expected_mu
        ss = sum((v - mean) ** 2 for v in col)
        if B > 1:
            expected_sd = (ss / (B - 1.0)) ** 0.5
        else:
            expected_sd = 0.0
        assert abs(vote_share[j] - expected_sd) < 1e-12

    # estimate is documented to be y_pred[0].
    assert result["estimate"] == y_pred[0]
    assert result["B"] == 5
    assert result["m"] == 3
    assert result["kind"] == "regression"


def test_btbg_edge():
    """Test edge cases."""
    rng_m = np.random.default_rng(42)
    rng_x = np.random.default_rng(42)
    # Faithful to the documented contract: models is B-by-m.
    models = rng_m.normal(0, 1, (5, 3))
    X_new = [rng_x.normal(0, 1) for _ in range(3)]
    kind = "regression"
    result = boot_bagging_predict(models, X_new, kind)
    assert isinstance(result, dict)
    assert "y_pred" in result
    assert "vote_share" in result
