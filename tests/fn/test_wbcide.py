"""Tests for wbcide.wooldridge_bjs_estimator."""

import math

import pytest

from morie.fn.wbcide import wooldridge_bjs_estimator

INF = float("inf")
# 9 units observed over 8 periods; three never-treated, three treated
# from t = 3 and three from t = 5.  A balanced panel is required.
COHORT = [3.0, 3.0, 3.0, 5.0, 5.0, 5.0, INF, INF, INF]
N_UNITS, N_PERIODS = len(COHORT), 8


def _panel():
    unit, time, g = [], [], []
    for i in range(N_UNITS):
        for t in range(N_PERIODS):
            unit.append(i)
            time.append(t)
            g.append(COHORT[i])
    D = [1.0 if t >= gv else 0.0 for t, gv in zip(time, g)]
    return unit, time, g, D


def _treated_cells():
    return [(gv, t) for gv in (3.0, 5.0) for t in range(int(gv), N_PERIODS)]


def test_wbcide_basic():
    """The interaction coefficients ARE the ATT(g,s) that generated y."""
    unit, time, g, D = _panel()

    # heterogeneous effect: 1.0 at adoption, growing 0.5 per period after
    def tau(gv, t):
        return 1.0 + 0.5 * (t - gv)

    y = [0.3 * u + 0.2 * t + (tau(gv, t) if d else 0.0)
         for u, t, gv, d in zip(unit, time, g, D)]

    res = wooldridge_bjs_estimator(y, D, unit, time)

    cells = _treated_cells()
    assert res["n_interactions"] == len(cells) == 8
    assert res["n_observations"] == N_UNITS * N_PERIODS
    assert res["min_cell_size"] == 3

    # every saturated coefficient recovers its own cell's true effect
    att = res["att_gt"]
    for gv, t in cells:
        assert att[(gv, t)] == pytest.approx(tau(gv, t), abs=1e-9)

    # the headline is the equally weighted mean over treated cells,
    # since every cohort-period cell here holds the same three units
    expected = sum(tau(gv, t) for gv, t in cells) / len(cells)
    assert res["estimate"] == pytest.approx(expected, abs=1e-9)

    # event-time aggregation: cells at the same horizon average together
    for r, value in res["event"].items():
        same = [tau(gv, t) for gv, t in cells if t - gv == r]
        assert value == pytest.approx(sum(same) / len(same), abs=1e-9)
    assert res["event"][0.0] == pytest.approx(1.0, abs=1e-9)
    assert res["event"][2.0] == pytest.approx(2.0, abs=1e-9)

    # cohort aggregation: the early cohort is observed at longer horizons
    # and therefore averages higher under this growing effect
    for gv, value in res["cohort_att"].items():
        same = [tau(gv, t) for gg, t in cells if gg == gv]
        assert value == pytest.approx(sum(same) / len(same), abs=1e-9)
    assert res["cohort_att"][3.0] > res["cohort_att"][5.0]

    # exact fit, so no residual variation and no sampling error
    assert res["se"] == pytest.approx(0.0, abs=1e-8)
    lo, hi = res["ci"]
    assert lo == pytest.approx(res["estimate"] - 1.959963984540054 * res["se"],
                               abs=1e-12)
    assert hi == pytest.approx(res["estimate"] + 1.959963984540054 * res["se"],
                               abs=1e-12)

    # with no covariates it is numerically the imputation estimator
    assert res["matches_imputation"] == pytest.approx(0.0, abs=1e-8)


def test_wbcide_with_covariates_recovers_the_common_slope():
    """A time-varying covariate enters with one common coefficient."""
    unit, time, g, D = _panel()
    x = [0.1 * ((7 * u + 3 * t) % 5) for u, t in zip(unit, time)]
    gamma = 1.5
    y = [0.3 * u + 0.2 * t + 2.0 * d + gamma * xv
         for u, t, d, xv in zip(unit, time, D, x)]

    res = wooldridge_bjs_estimator(y, D, unit, time, [[v] for v in x])

    coef = res["covariate_coef"]
    assert len(coef) == 1
    assert float(coef[0]) == pytest.approx(gamma, abs=1e-9)
    assert res["estimate"] == pytest.approx(2.0, abs=1e-9)
    for gv, t in _treated_cells():
        assert res["att_gt"][(gv, t)] == pytest.approx(2.0, abs=1e-9)
    # the imputation equivalence is only claimed without covariates
    assert res["matches_imputation"] is None


def test_wbcide_edge():
    """A homogeneous effect, a single cohort, and the balance requirement."""
    unit, time, g, D = _panel()
    y = [0.3 * u + 0.2 * t + 2.0 * d for u, t, d in zip(unit, time, D)]
    res = wooldridge_bjs_estimator(y, D, unit, time)
    assert res["estimate"] == pytest.approx(2.0, abs=1e-9)
    for value in res["att_gt"].values():
        assert value == pytest.approx(2.0, abs=1e-9)
    for value in res["event"].values():
        assert value == pytest.approx(2.0, abs=1e-9)
    assert all(math.isfinite(v) for v in res["ci"])

    # one treated cohort plus a never-treated group: a single event study
    unit2, time2, D2, y2 = [], [], [], []
    for i in range(6):
        gv = 2.0 if i < 3 else INF
        for t in range(4):
            d = 1.0 if t >= gv else 0.0
            unit2.append(i)
            time2.append(t)
            D2.append(d)
            y2.append(1.0 * i - 0.5 * t + 3.0 * d)
    r2 = wooldridge_bjs_estimator(y2, D2, unit2, time2)
    assert r2["n_interactions"] == 2          # t = 2 and t = 3
    assert r2["estimate"] == pytest.approx(3.0, abs=1e-9)
    assert set(r2["cohort_att"]) == {2.0}

    # an unbalanced panel is refused rather than silently differenced
    with pytest.raises(ValueError):
        wooldridge_bjs_estimator(y[:-1], D[:-1], unit[:-1], time[:-1])


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.wbcide as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
