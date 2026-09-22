"""Tests for evstud.event_study_coefficients."""

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

from morie.fn.evstud import event_study_coefficients


def test_evstud_basic():
    """Test basic functionality."""
    n = 100
    unit = np.concatenate([np.repeat(np.array([0, 1, 2, 3]), 25)])
    time = np.tile(np.linspace(1, 25, 25), 4)
    cohort = np.full(n, float("nan"))
    cohort[unit == 0] = 11.0
    cohort[unit == 1] = 11.0
    D = (time >= cohort).astype(float)
    D[unit == 2] = 0.0
    D[unit == 3] = 0.0

    # Impose parameters on the two-way FE + event-time specification
    alpha = np.array([0.5, -0.2, 0.1, -0.4])
    beta_t = np.linspace(0, 1, 25)
    mu_e = {2: 0.5, 3: 0.9, 5: 1.0, 7: 0.7}

    g = cohort
    e = np.where(np.isnan(g), -999, time - g)
    feats = mu_e[2] * (e == 2).astype(float) \
        + mu_e[3] * (e == 3).astype(float) \
        + mu_e[5] * (e == 5).astype(float) \
        + mu_e[7] * (e == 7).astype(float)
    y = alpha[unit.astype(int)] + beta_t[time.astype(int) - 1] + feats
    y = y + np.random.default_rng(0).normal(0, 0.1, n)

    result = event_study_coefficients(
        y, D, unit, time, cohort, max_lead=10, max_lag=10, ref=-1
    )

    assert isinstance(result, dict)
    expected_keys = {
        "estimate", "event_times", "coef", "se", "sigma2",
        "resid_df", "n_units", "n_periods", "n", "method",
    }
    assert set(result.keys()) == expected_keys
    # No "statistic" key exists
    assert "statistic" not in result

    coef = np.array(result["coef"])
    ets = np.array(result["event_times"])
    # Coefficients at e != 0 should match mu_e up to sampling noise (sigma=0.1)
    for e_val, mu in mu_e.items():
        idx = np.where(ets == e_val)[0][0]
        assert abs(coef[idx] - mu) < 0.3

    # Method string
    assert result["method"] == "Event-study leads + lags coefficients"

    # Custom OLS by hand using the design column for e=0 should match estimate
    off_index_0 = list(ets).index(0)
    assert abs(coef[off_index_0]) < 0.1


def test_evstud_edge():
    """Test edge cases for event_study_coefficients."""
    n = 100
    unit = np.concatenate([np.repeat(np.array([0, 1, 2, 3]), 25)])
    time = np.tile(np.linspace(1, 25, 25), 4)
    cohort = np.full(n, float("nan"))
    cohort[unit == 0] = 11.0
    cohort[unit == 1] = 11.0
    D = (time >= cohort).astype(float)
    D[unit == 2] = 0.0
    D[unit == 3] = 0.0

    g = cohort
    e = np.where(np.isnan(g), -999, time - g)
    feats = 0.5 * (e == 2).astype(float) + 0.9 * (e == 3).astype(float)
    y = np.zeros(n) + feats
    y = y + np.random.default_rng(1).normal(0, 0.1, n)

    result = event_study_coefficients(
        y, D, unit, time, cohort, max_lead=5, max_lag=5, ref=-1
    )

    assert isinstance(result, dict)
    # max_lead/max_lag must truncate event_times window
    assert max(result["event_times"]) <= 5
    assert min(result["event_times"]) >= -5

    # Lengths are consistent
    assert len(result["coef"]) == len(result["se"])
    assert len(result["coef"]) == len(result["event_times"])
