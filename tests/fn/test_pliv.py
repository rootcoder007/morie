"""Tests for morie.fn.pliv — Partially Linear IV / 2SLS LATE estimator."""

import math

import pytest

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd
from morie.fn.pliv import estimate_pliv


@pytest.fixture()
def iv_data():
    """Synthetic IV data with instrument Z, endogenous D, outcome Y."""
    rng = np.random.default_rng(42)
    n = 200
    x1 = rng.standard_normal(n)
    # Instrument: correlated with D but not directly with Y
    z = rng.standard_normal(n)
    # Endogenous treatment: depends on Z and X and unobservable u
    u = rng.standard_normal(n)
    d = 0.5 * z + 0.3 * x1 + 0.6 * u + rng.standard_normal(n) * 0.3
    # Outcome: depends on D, X, and u (confounded)
    y = 2.0 * d + 0.5 * x1 + 0.8 * u + rng.standard_normal(n) * 0.5
    return pd.DataFrame({"y": y, "d": d, "z": z, "x1": x1})


def test_returns_dict_with_keys(iv_data):
    """estimate_pliv returns a dict with all required keys."""
    result = estimate_pliv(
        iv_data,
        treatment="d",
        outcome="y",
        instrument="z",
        covariates=["x1"],
    )
    assert isinstance(result, dict)
    for key in ("late", "se", "ci_lower", "ci_upper", "pval", "n_obs", "method"):
        assert key in result, f"Missing key: {key}"


def test_late_is_finite(iv_data):
    """LATE and SE are finite floats."""
    result = estimate_pliv(
        iv_data,
        treatment="d",
        outcome="y",
        instrument="z",
        covariates=["x1"],
    )
    assert math.isfinite(result["late"])
    assert math.isfinite(result["se"])
    assert result["se"] > 0


def test_method_string(iv_data):
    """Method field should be a non-empty string."""
    result = estimate_pliv(
        iv_data,
        treatment="d",
        outcome="y",
        instrument="z",
        covariates=["x1"],
    )
    assert isinstance(result["method"], str)
    assert len(result["method"]) > 0


def test_missing_column_raises():
    """Missing columns should raise ValueError."""
    df = pd.DataFrame({"y": [1], "d": [0], "z": [1]})
    with pytest.raises(ValueError, match="Columns missing"):
        estimate_pliv(
            df,
            treatment="d",
            outcome="y",
            instrument="z",
            covariates=["nonexistent"],
        )


def test_ci_contains_late(iv_data):
    """95% CI should contain the point estimate."""
    result = estimate_pliv(
        iv_data,
        treatment="d",
        outcome="y",
        instrument="z",
        covariates=["x1"],
    )
    assert result["ci_lower"] <= result["late"] <= result["ci_upper"]


def test_pliv_without_covariates_is_the_cross_fitted_iv_ratio():
    """No covariates: l, m, r are fold-complement means; theta = sum(w u) / sum(w v)."""
    from morie.fn._rng import random_uniform

    z = [0.3, -1.2, 0.8, 1.5, -0.4, 0.9, -0.7, 0.1, 1.1, -1.6, 0.5, 0.2, -0.9, 1.3]
    d = [0.5, -0.8, 1.1, 1.2, 0.1, 0.4, -0.9, 0.6, 1.4, -1.0, 0.2, 0.7, -0.3, 0.9]
    y = [1.2, -1.5, 2.4, 2.1, 0.5, 0.6, -1.9, 1.0, 3.1, -2.2, 0.1, 1.6, -0.2, 1.7]
    n, K = len(z), 4
    r = estimate_pliv(
        pd.DataFrame({"z": z, "d": d, "y": y}),
        treatment="d",
        outcome="y",
        instrument="z",
        covariates=[],
        n_folds=K,
        random_state=7,
    )
    u = [float(v) for v in random_uniform(n, seed=7)]
    order = sorted(range(n), key=lambda i: (u[i], i))
    fold_of = {i: f for f in range(K) for i in order[f::K]}

    def cf(v):
        out = []
        for i in range(n):
            tr = [v[j] for j in range(n) if fold_of[j] != fold_of[i]]
            out.append(v[i] - sum(tr) / len(tr))
        return out

    uu, ww, vv = cf(y), cf(z), cf(d)
    theta = sum(a * b for a, b in zip(ww, uu)) / sum(a * b for a, b in zip(ww, vv))
    psi = [(a - theta * b) * c for a, b, c in zip(uu, vv, ww)]
    J = sum(a * b for a, b in zip(ww, vv)) / n
    assert r["late"] == pytest.approx(theta, rel=1e-12)
    assert r["se"] == pytest.approx(math.sqrt(sum(p * p for p in psi) / n / J**2 / n), rel=1e-12)
