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
    assert coint["group_adf"] < spur["group_adf"]
    assert coint["group_adf"] < 0


def test_table2_values_match_the_published_table():
    """Spot-checks against Pedroni (1999) Table 2 as printed."""
    from morie.fn.pdcoin import _PEDRONI_T2

    assert _PEDRONI_T2["standard"][2]["panel_v"] == (6.982, 81.145)
    assert _PEDRONI_T2["standard"][7]["group_t"] == (-3.723, 0.530)
    assert _PEDRONI_T2["intercept"][2]["group_t"] == (-2.453, 0.618)
    assert _PEDRONI_T2["trend"][7]["panel_rho"] == (-32.756, 154.378)
    # Every block covers m = 2..7 and five statistics; there is no m = 1.
    for case, block in _PEDRONI_T2.items():
        assert sorted(block) == [2, 3, 4, 5, 6, 7], case
        for m, row in block.items():
            assert set(row) == {"panel_v", "panel_rho", "panel_t", "group_rho", "group_t"}
            assert all(v > 0 for _, v in row.values()), "variances must be positive"


def test_m1_panel_refuses_to_standardise():
    """Z = (stat - mu sqrt(N)) / sqrt(v), Pedroni eq. (2), Group t column."""
    # This panel has a single regressor, and Table 2 has no m = 1 row,
    # so the standardisation must be refused rather than extrapolated.
    res = pedroni_panel_cointegration(*_panel(seed=9, N=8, cointegrated=True))
    assert res["n_regressors"] == 1
    assert res["z"] == {}
    assert res["p_values"] == {}
    assert any("m = 2..7" in w for w in res["warnings"])


def test_two_regressor_panel_gets_a_real_z_and_p():
    rng = np.random.default_rng(11)
    rows, grp = [], []
    for i in range(8):
        x1 = np.cumsum(rng.normal(0, 1, 120))
        x2 = np.cumsum(rng.normal(0, 1, 120))
        y = x1 + 0.5 * x2 + rng.normal(0, 1, 120)   # cointegrated
        rows.append(np.column_stack([y, x1, x2]))
        grp.append(np.full(120, i))
    res = pedroni_panel_cointegration(np.vstack(rows), np.concatenate(grp))
    assert res["n_regressors"] == 2
    assert set(res["z"]) == {"panel_v", "panel_rho", "panel_t", "group_rho", "group_t"}
    assert res["p_value"] < 0.01, "a strongly cointegrated panel should reject"


def test_cointegrated_panel_has_rho_further_below_zero():
    coint = pedroni_panel_cointegration(*_panel(seed=2, cointegrated=True))
    spur = pedroni_panel_cointegration(*_panel(seed=2, cointegrated=False))
    assert coint["panel_rho"] < spur["panel_rho"]


def test_group_statistic_is_the_mean_of_the_per_unit_ones():
    res = pedroni_panel_cointegration(*_panel(seed=3, cointegrated=True))
    assert res["mean_adf_t"] == pytest.approx(np.nanmean(res["per_unit_adf"]))


def test_group_adf_is_the_root_n_scaling_of_the_mean_t():
    """Pedroni's Table 1 group t form is N^-1/2 sum_i t_i."""
    res = pedroni_panel_cointegration(*_panel(seed=4, N=6, cointegrated=True))
    finite = res["per_unit_adf"][np.isfinite(res["per_unit_adf"])]
    assert res["group_adf"] == pytest.approx(res["mean_adf_t"] * np.sqrt(finite.size))


def _mpanel(seed=0, N=12, T=100, coint=True, m=2):
    """Panel with m regressors, so Table 2 applies."""
    rng = np.random.default_rng(seed)
    rows, grp = [], []
    for i in range(N):
        xs = [np.cumsum(rng.normal(0, 1, T)) for _ in range(m)]
        y = sum(xs) + rng.normal(0, 1, T) if coint else np.cumsum(rng.normal(0, 1, T))
        rows.append(np.column_stack([y] + xs))
        grp.append(np.full(T, i))
    return np.vstack(rows), np.concatenate(grp)


def test_all_five_statistics_reject_when_cointegrated():
    res = pedroni_panel_cointegration(*_mpanel(seed=3, coint=True))
    for name in ("panel_v", "panel_rho", "panel_t", "group_rho", "group_t"):
        assert res["p_values"][name] < 0.01, name


def test_no_statistic_rejects_a_spurious_panel():
    res = pedroni_panel_cointegration(*_mpanel(seed=3, coint=False))
    for name in ("panel_v", "panel_rho", "panel_t", "group_rho", "group_t"):
        assert res["p_values"][name] > 0.05, name


def test_panel_v_is_the_right_tail_statistic():
    """Pedroni: the panel variance statistic diverges to +infinity under
    the alternative, so large positive values reject. The other four go
    negative. This pins the tail conventions."""
    coint = pedroni_panel_cointegration(*_mpanel(seed=3, coint=True))
    assert coint["z"]["panel_v"] > 0
    for name in ("panel_rho", "panel_t", "group_rho", "group_t"):
        assert coint["z"][name] < 0, name


def test_standardisation_is_pedroni_equation_2():
    """Z = (stat - mu sqrt(N)) / sqrt(v), read straight off Table 2."""
    from morie.fn.pdcoin import _PEDRONI_T2

    res = pedroni_panel_cointegration(*_mpanel(seed=7, N=10, coint=True))
    for name, (mu, v) in _PEDRONI_T2["intercept"][2].items():
        want = (res["statistics"][name] - mu * np.sqrt(res["n_units"])) / np.sqrt(v)
        assert res["z"][name] == pytest.approx(want, rel=1e-12), name


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
