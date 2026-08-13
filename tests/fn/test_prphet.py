"""Tests for the Prophet trio. Full anchor: wave3/anchor_prophet.py."""
import math
import pytest
from morie.fn import _s03core as k
from morie.fn.prnFil import (changepoint_path, select_changepoints,
                             trend_intervals)
from morie.fn.prophe import additive_components, component_shares
from morie.fn.prphet import (fourier_terms, holiday_matrix,
                             piecewise_trend, prophet_fit,
                             prophet_predict)

N = 120
T = [float(i) for i in range(N)]
TRUE_CP = 60.0
SEAS = [("yearly", 12.0, 3)]
HOL = {"event": [30.0, 90.0]}
Y = [(2.0 + 0.15 * i + (0.4 * (i - TRUE_CP) if i >= TRUE_CP else 0.0)
      + 3.0 * math.sin(2 * math.pi * i / 12.0)
      + (5.0 if i in (30, 90) else 0.0)) for i in range(N)]


def test_the_trend_is_continuous_at_every_changepoint():
    """gamma_j = -s_j delta_j is what joins the segments. Without it the
    curve jumps and least squares hides it in the residual."""
    cps = [10.0, 25.0, 40.0]
    d = [0.8, -1.5, 0.3]
    for s in cps:
        left = piecewise_trend([s - 1e-7], 0.5, 2.0, d, cps)[0]
        right = piecewise_trend([s + 1e-7], 0.5, 2.0, d, cps)[0]
        assert abs(left - right) < 1e-6


def test_the_fourier_basis_is_exactly_periodic():
    a = fourier_terms([3.0], 12.0, 3)[0]
    b = fourier_terms([15.0], 12.0, 3)[0]
    assert a == pytest.approx(b, abs=1e-12)
    assert len(fourier_terms([0.0], 12.0, 5)[0]) == 10
    with pytest.raises(ValueError):
        fourier_terms([0.0], 0.0, 3)
    with pytest.raises(ValueError):
        fourier_terms([0.0], 12.0, 0)


def test_holidays_get_their_own_widened_indicator():
    hm, names = holiday_matrix([9.0, 10.0, 11.0, 20.0],
                               {"launch": [10.0]}, lower=1, upper=1)
    assert [r[0] for r in hm] == [1.0, 1.0, 1.0, 0.0]
    assert names == ["launch"]


def test_the_fit_recovers_the_components():
    f = prophet_fit(T, Y, n_changepoints=12, seasonalities=SEAS,
                    holidays=HOL, changepoint_prior=0.5)
    rmse = math.sqrt(k.mean([v * v for v in f["residual"]]))
    assert rmse < 0.05 * (max(Y) - min(Y))
    assert abs(f["coef"]["holiday_event"] - 5.0) < 1.5
    pred = prophet_predict(f, [120.0, 121.0], seasonalities=SEAS,
                           holidays=HOL)
    assert len(pred) == 2 and pred[1] != pred[0]
    with pytest.raises(ValueError):
        prophet_predict(f, [120.0])
    with pytest.raises(ValueError):
        prophet_fit(T, Y, changepoint_prior=0.0)


def test_the_decomposition_reconstructs_exactly():
    """A decomposition that does not sum back is a picture of something
    else -- and no error metric would reveal it."""
    d = additive_components(T, Y, seasonalities=SEAS, holidays=HOL,
                            n_changepoints=12, changepoint_prior=0.5)
    assert d["reconstructs"]
    assert d["reconstruction_error"] < 1e-9
    assert set(d["component_names"]) == {"trend", "yearly", "holidays"}
    # dropping a component must break it, or the check is vacuous
    part = dict(d["components"])
    part.pop("yearly")
    tot = [sum(part[c][i] for c in part) for i in range(N)]
    assert max(abs(tot[i] - d["fitted"][i]) for i in range(N)) > 0.5
    sh = component_shares(d["components"])
    assert sh["ranked"][0] == "trend"
    assert "not an orthogonal" in sh["note"]


def test_tau_trades_training_error_against_flexibility():
    """The paper's overfitting signal, and it needs GENUINE L1 -- a
    ridge shrinks every delta and zeroes none, so nothing is selected."""
    path = changepoint_path(T, Y, taus=[0.001, 0.01, 0.1, 1.0],
                            n_changepoints=15, seasonalities=SEAS)
    for i in range(len(path) - 1):
        assert path[i + 1]["rmse"] <= path[i]["rmse"] + 1e-9
    assert path[-1]["active"] > path[0]["active"]
    assert path[0]["active"] < 15          # sparsity is real
    assert path[-1]["l1"] > path[0]["l1"]
    with pytest.raises(ValueError):
        changepoint_path(T, Y, taus=[0.1])


def test_selection_finds_the_true_break_and_stays_sparse():
    s = select_changepoints(T, Y, tau=0.05, n_changepoints=15,
                            seasonalities=SEAS)
    assert s["n_selected"] < s["n_candidates"]
    assert any(abs(c - TRUE_CP) < 10.0 for c in s["selected"])
    # candidates stop at 80% -- one near the end has no data after it
    assert s["last_candidate_fraction"] <= 0.81


def test_the_trend_interval_widens_with_horizon():
    s = select_changepoints(T, Y, tau=0.5, n_changepoints=15,
                            seasonalities=SEAS)
    iv = trend_intervals(s["fit"], [float(120 + h) for h in range(12)],
                         level=0.8, n_sims=80, seed=3)
    assert iv["width"][-1] >= iv["width"][0]
    assert "does not claim exact coverage" in iv["note"]
