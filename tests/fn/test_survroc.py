"""Time-dependent ROC curves for censored survival data."""
import importlib

import pytest

R = importlib.import_module("morie.fn.survroc")

TIMES = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
         11.0, 12.0]
EVENTS = [1] * 12
MARK = [9.0, 8.5, 4.0, 8.0, 6.0, 5.5, 7.0, 5.0, 3.0, 2.5, 2.0, 1.0]
HZ = 6.5
CE = [0 if i in (2, 5, 8) else 1 for i in range(12)]


def test_kaplan_meier_is_the_product_limit():
    T = [2.0, 3.0, 4.0, 4.0, 6.0, 8.0]
    E = [1, 0, 1, 1, 0, 1]
    assert R.kaplan_meier(T, E, 1.0) == pytest.approx(1.0)
    assert R.kaplan_meier(T, E, 2.0) == pytest.approx(5.0 / 6.0)
    assert R.kaplan_meier(T, E, 4.0) == pytest.approx(5.0 / 12.0)


def test_the_curve_is_a_step_function_starting_at_one():
    c = R.kaplan_meier(TIMES, EVENTS)
    assert c[0] == (0.0, 1.0)
    assert all(c[i][1] >= c[i + 1][1] for i in range(len(c) - 1))


@pytest.mark.parametrize("c", sorted(set(MARK)))
def test_km_and_empirical_agree_without_censoring(c):
    a = R._pair(TIMES, EVENTS, MARK, c, HZ, "km")
    b = R._pair(TIMES, EVENTS, MARK, c, HZ, "empirical")
    assert a[0] == pytest.approx(b[0], abs=1e-12)
    assert a[1] == pytest.approx(b[1], abs=1e-12)


def test_the_auc_is_the_mann_whitney_statistic():
    case = [MARK[i] for i in range(12) if TIMES[i] <= HZ]
    ctrl = [MARK[i] for i in range(12) if TIMES[i] > HZ]
    u = sum(1.0 if a > b else 0.5 if a == b else 0.0
            for a in case for b in ctrl)
    auc = R.auc_at(TIMES, EVENTS, MARK, HZ)
    assert auc == pytest.approx(u / (len(case) * len(ctrl)),
                                abs=1e-12)
    assert 0.5 < auc < 1.0


def test_markers_with_known_areas():
    assert R.auc_at(TIMES, EVENTS, [-t for t in TIMES], HZ) \
        == pytest.approx(1.0)
    assert R.auc_at(TIMES, EVENTS, TIMES, HZ) == pytest.approx(0.0)
    assert R.auc_at(TIMES, EVENTS, [3.0] * 12, HZ) \
        == pytest.approx(0.5)


def test_reversing_the_marker_reflects_the_area():
    a = R.auc_at(TIMES, EVENTS, MARK, HZ)
    b = R.auc_at(TIMES, EVENTS, [-m for m in MARK], HZ)
    assert b == pytest.approx(1.0 - a, abs=1e-12)


def test_the_curve_is_monotone_in_the_threshold():
    pts = sorted(R.roc_at(TIMES, EVENTS, MARK, HZ),
                 key=lambda p: p["threshold"])
    se = [p["sensitivity"] for p in pts]
    sp = [p["specificity"] for p in pts]
    assert all(se[i] >= se[i + 1] - 1e-12 for i in range(len(se) - 1))
    assert all(sp[i] <= sp[i + 1] + 1e-12 for i in range(len(sp) - 1))


def test_the_curve_spans_the_unit_square():
    pts = sorted(R.roc_at(TIMES, EVENTS, MARK, HZ),
                 key=lambda p: p["threshold"])
    assert pts[0]["sensitivity"] == pytest.approx(1.0)
    assert pts[0]["specificity"] == pytest.approx(0.0)
    assert pts[-1]["sensitivity"] == pytest.approx(0.0)
    assert pts[-1]["specificity"] == pytest.approx(1.0)


def test_the_area_depends_on_the_horizon():
    assert R.auc_at(TIMES, EVENTS, MARK, 3.5) \
        != pytest.approx(R.auc_at(TIMES, EVENTS, MARK, 9.5))


def test_the_km_route_handles_censoring():
    a = R.auc_at(TIMES, CE, MARK, HZ)
    assert abs(a - R.auc_at(TIMES, EVENTS, MARK, HZ)) < 0.15


def test_the_empirical_route_refuses_censored_data():
    with pytest.raises(ValueError):
        R.auc_at(TIMES, CE, MARK, HZ, "empirical")


def test_the_result_accounts_for_every_subject():
    r = R.time_dependent_roc(TIMES, CE, MARK, HZ)
    assert r["n"] == 12
    assert r["n_events_by_t"] + r["n_at_risk_after_t"] \
        + r["n_censored_before_t"] == 12
    assert 0.0 < r["survival_at_t"] < 1.0
    assert r["route"] == "km"


@pytest.mark.parametrize("call", [
    lambda: R.auc_at(TIMES, EVENTS[:5], MARK, HZ),
    lambda: R.auc_at(TIMES, EVENTS, MARK[:5], HZ),
    lambda: R.auc_at([], [], [], HZ),
    lambda: R.auc_at(TIMES, [2] * 12, MARK, HZ),
    lambda: R.auc_at([-1.0] + TIMES[1:], EVENTS, MARK, HZ),
    lambda: R.auc_at(TIMES, EVENTS, MARK, 0.0),
    lambda: R.auc_at(TIMES, EVENTS, MARK, 0.5),
    lambda: R.auc_at(TIMES, EVENTS, MARK, 20.0),
    lambda: R.auc_at(TIMES, EVENTS, MARK, HZ, "smooth"),
])
def test_bad_input_is_refused(call):
    with pytest.raises(ValueError):
        call()
