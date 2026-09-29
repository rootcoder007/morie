"""Tests for morie.fn.aipw — Augmented IPW doubly-robust ATE estimator."""

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd
import math

import pytest

from morie.fn.aipw import estimate_aipw


@pytest.fixture()
def synth_data():
    rng = np.random.default_rng(42)
    n = 200
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    prob = 1 / (1 + np.exp(-(0.4 * x1 - 0.2 * x2)))
    t = rng.binomial(1, prob)
    y = (0.3 * t + 0.2 * x1 + rng.standard_normal(n) * 0.5 > 0.2).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "treatment": t, "outcome": y})


def test_returns_dict_with_ate(synth_data):
    result = estimate_aipw(synth_data, treatment="treatment", outcome="outcome", covariates=["x1", "x2"])
    assert isinstance(result, dict)
    assert "ate" in result


def test_ate_is_finite(synth_data):
    result = estimate_aipw(synth_data, treatment="treatment", outcome="outcome", covariates=["x1", "x2"])
    assert np.isfinite(result["ate"])
    assert np.isfinite(result["se"])


def test_ci_brackets_ate(synth_data):
    result = estimate_aipw(synth_data, treatment="treatment", outcome="outcome", covariates=["x1", "x2"])
    assert result["ci_lower"] <= result["ate"] <= result["ci_upper"]


def test_has_expected_keys(synth_data):
    result = estimate_aipw(synth_data, treatment="treatment", outcome="outcome", covariates=["x1", "x2"])
    for key in ("ate", "se", "ci_lower", "ci_upper", "n", "method"):
        assert key in result


def test_linear_outcome_model(synth_data):
    result = estimate_aipw(
        synth_data,
        treatment="treatment",
        outcome="outcome",
        covariates=["x1", "x2"],
        outcome_model="linear",
    )
    assert np.isfinite(result["ate"])


def _logit2(x, y):
    """Maximum-likelihood logistic regression of y on (1, x) by Newton's method."""
    b0 = b1 = 0.0
    for _ in range(100):
        p = [1 / (1 + math.exp(-(b0 + b1 * v))) for v in x]
        g0 = sum(a - q for a, q in zip(y, p))
        g1 = sum(v * (a - q) for v, a, q in zip(x, y, p))
        w = [q * (1 - q) for q in p]
        h00, h01, h11 = sum(w), sum(a * v for a, v in zip(w, x)), sum(a * v * v for a, v in zip(w, x))
        det = h00 * h11 - h01 * h01
        s0, s1 = (h11 * g0 - h01 * g1) / det, (h00 * g1 - h01 * g0) / det
        b0, b1 = b0 + s0, b1 + s1
        if max(abs(s0), abs(s1)) < 1e-13:
            break
    return b0, b1


def test_aipw_linear_recomputed_from_its_parts():
    """ps by logistic MLE on (1, x) clipped to [0.01, 0.99]; arm-wise OLS outcome
    regressions; psi = mu1 - mu0 + T(Y - mu1)/e - (1 - T)(Y - mu0)/(1 - e)."""
    import math

    t = [0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0]
    x = [0.2, 1.1, -0.5, 0.9, 1.4, 0.1, 0.3, -1.0, 2.0, 0.6, -0.2, 0.8]
    y = [1.1, 3.4, 0.2, 3.0, 3.9, 1.0, 2.5, -0.3, 4.6, 1.6, 2.0, 1.9]
    b0, b1 = _logit2(x, t)
    e = [min(max(1 / (1 + math.exp(-(b0 + b1 * v))), 0.01), 0.99) for v in x]

    def arm(k):
        xs = [a for a, c in zip(x, t) if c == k]
        ys = [a for a, c in zip(y, t) if c == k]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        s = sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / sum((a - mx) ** 2 for a in xs)
        return [my + s * (v - mx) for v in x]

    mu1, mu0 = arm(1), arm(0)
    psi = [m1 - m0 + c * (yy - m1) / p - (1 - c) * (yy - m0) / (1 - p) for m1, m0, c, yy, p in zip(mu1, mu0, t, y, e)]
    n = len(psi)
    ate = sum(psi) / n
    se = math.sqrt(sum((v - ate) ** 2 for v in psi) / (n - 1) / n)
    r = estimate_aipw(pd.DataFrame({"t": t, "x": x, "y": y}), treatment="t", outcome="y", covariates=["x"],
                      outcome_model="linear")
    assert r["ate"] == pytest.approx(ate, rel=1e-9)
    assert r["se"] == pytest.approx(se, rel=1e-9)
