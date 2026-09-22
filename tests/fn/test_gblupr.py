"""Tests for gblupr.gblup_estimator."""

from morie.fn import _array_core as np

from morie.fn.gblupr import gblup_estimator


def _to_list(arr):
    """Convert a numpy array (1D or 2D) to a plain Python list."""
    return arr.tolist()


def test_gblupr_basic():
    """Test basic functionality."""
    rng_y = np.random.default_rng(43)
    y = rng_y.normal(0, 1, 100).tolist()
    X = np.random.default_rng(42).normal(0, 1, (100, 5)).tolist()
    Z = np.random.default_rng(43).normal(0, 1, (100, 10)).tolist()
    G = np.eye(10).tolist()
    result = gblup_estimator(y, X, Z, G)
    # Result is a RichResult (dict-like); the documented keys live in .payload
    assert hasattr(result, "payload")
    payload = result.payload if not isinstance(result, dict) else result
    # Verify expected keys are present
    for key in ("beta", "u", "fitted", "residual_ss", "lambda", "n", "method"):
        assert key in payload
    # Independent recomputation: lambda = var_e / var_u (defaults 1.0 each)
    assert payload["lambda"] == 1.0
    # fitted[i] = X[i] @ beta + Z[i] @ u
    beta = payload["beta"]
    u = payload["u"]
    fitted_expected = [
        sum(X[i][a] * beta[a] for a in range(len(beta)))
        + sum(Z[i][c] * u[c] for c in range(len(u)))
        for i in range(len(y))
    ]
    for a, b in zip(payload["fitted"], fitted_expected):
        assert abs(a - b) < 1e-8


def test_gblupr_edge():
    """Test edge cases (defaults, positive variance, simple G=I)."""
    y = np.random.default_rng(43).normal(0, 1, 100).tolist()
    X = np.random.default_rng(42).normal(0, 1, (100, 5)).tolist()
    Z = np.random.default_rng(43).normal(0, 1, (100, 10)).tolist()
    G = np.eye(10).tolist()
    result = gblup_estimator(y, X, Z, G)
    payload = result.payload if not isinstance(result, dict) else result
    assert payload["n"] == 100
    assert len(payload["u"]) == 10
    assert len(payload["beta"]) == 5
    assert isinstance(payload["method"], str)
