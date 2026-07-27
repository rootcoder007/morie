"""Tests for pdcoin.pedroni_panel_cointegration."""

import numpy as np
import pytest

from morie.fn.pdcoin import pedroni_panel_cointegration


def _panel(seed=0, N=8, T=120, cointegrated=False):
    rng = np.random.default_rng(seed)
    rows, grp = [], []
    for i in range(N):
        x = np.cumsum(rng.normal(0, 1, T))
        y = x + rng.normal(0, 1, T) if cointegrated else np.cumsum(rng.normal(0, 1, T))
        rows.append(np.column_stack([y, x]))
        grp.append(np.full(T, i))
    return np.vstack(rows), np.concatenate(grp)


def test_cointegrated_panel_gives_a_more_negative_adf():
    """Stationary residuals mean-revert, so the ADF t goes negative."""
    coint = pedroni_panel_cointegration(*_panel(seed=1, cointegrated=True))
    spur = pedroni_panel_cointegration(*_panel(seed=1, cointegrated=False))
    assert coint["panel_adf"] < spur["panel_adf"]
    assert coint["panel_adf"] < 0


def test_cointegrated_panel_has_rho_further_below_zero():
    coint = pedroni_panel_cointegration(*_panel(seed=2, cointegrated=True))
    spur = pedroni_panel_cointegration(*_panel(seed=2, cointegrated=False))
    assert coint["panel_rho"] < spur["panel_rho"]


def test_group_statistic_is_the_mean_of_the_per_unit_ones():
    res = pedroni_panel_cointegration(*_panel(seed=3, cointegrated=True))
    assert res["group_adf"] == pytest.approx(np.nanmean(res["per_unit_adf"]))


def test_panel_statistic_is_the_root_n_scaling_of_the_group_one():
    res = pedroni_panel_cointegration(*_panel(seed=4, N=6, cointegrated=True))
    finite = res["per_unit_adf"][np.isfinite(res["per_unit_adf"])]
    assert res["panel_adf"] == pytest.approx(res["group_adf"] * np.sqrt(finite.size))


def test_no_p_value_without_simulation_or_cdf():
    """An unstandardised statistic must not be handed a normal p-value."""
    res = pedroni_panel_cointegration(*_panel(seed=5))
    assert res["p_value"] is None
    assert any("unstandardised" in w for w in res["warnings"])


def test_simulated_p_value_separates_the_two_cases():
    X, g = _panel(seed=6, N=6, T=100, cointegrated=True)
    coint = pedroni_panel_cointegration(X, g, nsim=40, seed=2)
    Xs, gs = _panel(seed=6, N=6, T=100, cointegrated=False)
    spur = pedroni_panel_cointegration(Xs, gs, nsim=40, seed=2)
    assert coint["p_value"] < spur["p_value"]


def test_units_too_short_are_reported_not_silently_dropped():
    X, g = _panel(seed=7, N=4, T=60)
    g[:3] = 99  # a 3-row unit cannot support the ADF regression
    res = pedroni_panel_cointegration(X, g)
    assert any("skipped" in w for w in res["warnings"])


def test_validates_inputs():
    X, g = _panel(seed=8, N=3, T=60)
    with pytest.raises(ValueError, match="at least one regressor"):
        pedroni_panel_cointegration(X[:, :1], g)
    with pytest.raises(ValueError, match="one entry per row"):
        pedroni_panel_cointegration(X, g[:-1])
    with pytest.raises(ValueError, match="at least 2 panel units"):
        pedroni_panel_cointegration(X, np.zeros_like(g))
    with pytest.raises(ValueError, match="must be finite"):
        bad = X.copy()
        bad[0, 0] = np.nan
        pedroni_panel_cointegration(bad, g)
    with pytest.raises(ValueError, match="lags must not be negative"):
        pedroni_panel_cointegration(X, g, lags=-1)
