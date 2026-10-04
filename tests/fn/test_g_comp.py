"""Tests for morie.fn.g_comp — G-computation ATE estimator with bootstrap SE."""

import math

import pytest

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd
from morie.fn.g_comp import estimate_ate_gcomputation


@pytest.fixture()
def gcomp_data():
    """Synthetic data with binary treatment and continuous outcome."""
    rng = np.random.default_rng(42)
    n = 200
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    t = (rng.uniform(size=n) < 0.5).astype(float)
    y = 1.0 + 1.5 * t + 0.8 * x1 - 0.3 * x2 + rng.standard_normal(n) * 0.5
    return pd.DataFrame({"y": y, "t": t, "x1": x1, "x2": x2})


def test_returns_dict_with_keys(gcomp_data):
    """estimate_ate_gcomputation returns dict with required keys."""
    result = estimate_ate_gcomputation(
        gcomp_data,
        treatment="t",
        outcome="y",
        covariates=["x1", "x2"],
    )
    assert isinstance(result, dict)
    for key in ("ate", "se", "ci_lower", "ci_upper", "n_obs", "outcome_model"):
        assert key in result


def test_ci_contains_ate(gcomp_data):
    """Bootstrap CI should contain the point estimate."""
    result = estimate_ate_gcomputation(
        gcomp_data,
        treatment="t",
        outcome="y",
        covariates=["x1", "x2"],
    )
    assert result["ci_lower"] <= result["ate"] <= result["ci_upper"]


def test_bootstrap_se_finite(gcomp_data):
    """Bootstrap SE should be finite and positive."""
    result = estimate_ate_gcomputation(
        gcomp_data,
        treatment="t",
        outcome="y",
        covariates=["x1", "x2"],
    )
    assert math.isfinite(result["se"])
    assert result["se"] > 0


def test_invalid_outcome_model_raises(gcomp_data):
    """Invalid outcome_model should raise ValueError."""
    with pytest.raises(ValueError, match="outcome_model"):
        estimate_ate_gcomputation(
            gcomp_data,
            treatment="t",
            outcome="y",
            covariates=["x1", "x2"],
            outcome_model="invalid",
        )


def test_too_few_observations_raises():
    """Fewer than 10 rows should raise ValueError."""
    df = pd.DataFrame(
        {
            "y": [1, 2, 3],
            "t": [0, 1, 0],
            "x1": [0.1, 0.2, 0.3],
        }
    )
    with pytest.raises(ValueError, match="at least 10"):
        estimate_ate_gcomputation(
            df,
            treatment="t",
            outcome="y",
            covariates=["x1"],
        )


GD = {
    "t": [0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0],
    "x": [0.2, 1.1, -0.5, 0.9, 1.4, 0.1, 0.3, -1.0, 2.0, 0.6, -0.2, 0.8],
    "y": [1.1, 3.4, 0.2, 3.0, 3.9, 1.0, 2.5, -0.3, 4.6, 1.6, 2.0, 1.9],
}


def _ols_t_coef(t, x, y):
    """Coefficient on t in y ~ 1 + t + x (Frisch-Waugh: residualise t and y on (1, x))."""
    n = len(t)
    mx = sum(x) / n

    def res(v):
        mv = sum(v) / n
        b = sum((a - mx) * (c - mv) for a, c in zip(x, v)) / sum((a - mx) ** 2 for a in x)
        return [c - mv - b * (a - mx) for a, c in zip(x, v)]

    rt, ry = res(t), res(y)
    return sum(a * b for a, b in zip(rt, ry)) / sum(a * a for a in rt)


def test_linear_gcomp_is_the_ols_treatment_coefficient_and_its_bootstrap():
    """With a main-effects linear model Y(1) - Y(0) is the T coefficient for every unit;
    the SE is the SD of that coefficient over the Philox bootstrap resamples."""
    from morie.fn._rng import random_uniform

    r = estimate_ate_gcomputation(pd.DataFrame(GD), treatment="t", outcome="y", covariates=["x"])
    assert r["ate"] == pytest.approx(_ols_t_coef(GD["t"], GD["x"], GD["y"]), rel=1e-10)
    n = len(GD["t"])
    reps = []
    for b in range(500):
        idx = [min(int(float(u) * n), n - 1) for u in list(random_uniform(n, seed=42, stream=b))]
        tb = [GD["t"][i] for i in idx]
        if len(set(tb)) < 2:
            continue
        reps.append(_ols_t_coef(tb, [GD["x"][i] for i in idx], [GD["y"][i] for i in idx]))
    m = sum(reps) / len(reps)
    sd = math.sqrt(sum((v - m) ** 2 for v in reps) / (len(reps) - 1))
    assert r["se"] == pytest.approx(sd, rel=1e-8)


def test_logistic_gcomp_uses_the_unpenalised_mle():
    yb = [1 if v > 1.8 else 0 for v in GD["y"]]
    yb[0], yb[3] = 1, 0  # break separation
    d = pd.DataFrame({"t": GD["t"], "x": GD["x"], "y": yb})
    r = estimate_ate_gcomputation(d, treatment="t", outcome="y", covariates=["x"], outcome_model="logistic")
    # Newton for y ~ 1 + t + x
    X = [[1.0, float(a), b] for a, b in zip(GD["t"], GD["x"])]
    beta = [0.0, 0.0, 0.0]
    from morie.fn._qpcore import solve

    for _ in range(100):
        p = [1 / (1 + math.exp(-sum(c * v for c, v in zip(beta, row)))) for row in X]
        g = [sum(row[a] * (yy - q) for row, yy, q in zip(X, yb, p)) for a in range(3)]
        H = [[sum(row[a] * row[c] * q * (1 - q) for row, q in zip(X, p)) for c in range(3)] for a in range(3)]
        s = solve(H, g)
        beta = [c + v for c, v in zip(beta, s)]
        if max(abs(v) for v in s) < 1e-13:
            break

    def pr(tv, xv):
        return 1 / (1 + math.exp(-(beta[0] + beta[1] * tv + beta[2] * xv)))

    ate = sum(pr(1, xv) - pr(0, xv) for xv in GD["x"]) / len(yb)
    assert r["ate"] == pytest.approx(ate, rel=1e-8)
