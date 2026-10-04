"""Tests for morie.fn.ate — IPW-weighted OLS ATE estimator."""

import math

import pytest

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd
from morie.fn.ate import estimate_ate


@pytest.fixture()
def ate_data():
    """Synthetic data with known treatment effect (~2.0) and IPW weights."""
    rng = np.random.default_rng(42)
    n = 200
    x = rng.standard_normal(n)
    t = (rng.uniform(size=n) < (1 / (1 + np.exp(-x)))).astype(float)
    y = 1.0 + 2.0 * t + 0.5 * x + rng.standard_normal(n) * 0.5
    # Simple IPW weights (inverse of propensity)
    ps = 1 / (1 + np.exp(-x))
    w = np.where(t == 1, 1 / ps, 1 / (1 - ps))
    return pd.DataFrame({"y": y, "t": t, "x": x, "w": w})


def test_returns_tuple(ate_data):
    """estimate_ate returns a (coef, se) tuple."""
    result = estimate_ate(ate_data, outcome="y", treatment="t", weights_col="w")
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_coef_is_finite(ate_data):
    """ATE coefficient and SE are finite floats."""
    coef, se = estimate_ate(ate_data, outcome="y", treatment="t", weights_col="w")
    assert math.isfinite(coef)
    assert math.isfinite(se)


def test_coef_near_true_effect(ate_data):
    """ATE coefficient should be roughly near the true effect of 2.0."""
    coef, se = estimate_ate(ate_data, outcome="y", treatment="t", weights_col="w")
    # Generous tolerance for n=200 with noise
    assert abs(coef - 2.0) < 2.0, f"ATE={coef} too far from 2.0"


def test_se_positive(ate_data):
    """Standard error must be strictly positive."""
    _, se = estimate_ate(ate_data, outcome="y", treatment="t", weights_col="w")
    assert se > 0


def test_weighted_ols_uses_hc3(ate_data):
    """Verify the function runs without error (HC3 robust covariance)."""
    coef, se = estimate_ate(ate_data, outcome="y", treatment="t", weights_col="w")
    # HC3 SEs are typically larger than non-robust SEs; just check it works
    assert se > 0


def test_ate_is_the_hajek_contrast_with_hc3_errors():
    """Coefficient = weighted mean difference; HC3 on sqrt(w)-scaled rows of [1, t]."""
    y = [1.0, 2.2, 1.7, 3.1, 2.8, 3.9, 0.4, 2.5]
    t = [0, 0, 0, 1, 1, 1, 0, 1]
    w = [1.2, 2.0, 1.5, 1.1, 3.0, 1.4, 2.6, 1.9]
    coef, se = estimate_ate(pd.DataFrame({"y": y, "t": t, "w": w}), outcome="y", treatment="t", weights_col="w")
    m1 = sum(a * b for a, b, c in zip(w, y, t) if c) / sum(a for a, c in zip(w, t) if c)
    m0 = sum(a * b for a, b, c in zip(w, y, t) if not c) / sum(a for a, c in zip(w, t) if not c)
    assert coef == pytest.approx(m1 - m0, rel=1e-12)
    # (X'WX)^-1 for X = [1, t] in closed form
    S, St = sum(w), sum(a * c for a, c in zip(w, t))
    det = S * St - St * St
    inv = [[St / det, -St / det], [-St / det, S / det]]
    b0 = m0
    meat = [[0.0, 0.0], [0.0, 0.0]]
    for wi, yi, ti in zip(w, y, t):
        x = [1.0, float(ti)]
        e = yi - b0 - coef * ti
        h = wi * sum(x[a] * inv[a][b] * x[b] for a in range(2) for b in range(2))
        for a in range(2):
            for b in range(2):
                meat[a][b] += wi * x[a] * x[b] * wi * e * e / (1 - h) ** 2
    v = sum(inv[1][a] * meat[a][b] * inv[b][1] for a in range(2) for b in range(2))
    assert se == pytest.approx(math.sqrt(v), rel=1e-10)
