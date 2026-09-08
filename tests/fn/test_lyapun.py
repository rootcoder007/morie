"""Tests for lyapun (Rosenstein, Collins & De Luca 1993)."""

import math

from morie.fn.lyapun import (autocorrelation_lag, divergence_curve, embed,
                             largest_lyapunov, lyapunov_exponent,
                             mean_period)


def _logistic(n, mu=4.0, burn=200):
    x = 0.1
    for _ in range(burn):
        x = mu * x * (1.0 - x)
    out = []
    for _ in range(n):
        x = mu * x * (1.0 - x)
        out.append(x)
    return out


def _henon(n, a=1.4, b=0.3, burn=500):
    x, y = 0.1, 0.1
    for _ in range(burn):
        x, y = 1.0 - a * x * x + y, b * x
    out = []
    for _ in range(n):
        x, y = 1.0 - a * x * x + y, b * x
        out.append(x)
    return out


def test_embedding_shape_and_contents():
    s = [float(t) for t in range(20)]
    pts = embed(s, 4, 3)
    assert len(pts) == 20 - 3 * 3          # M = N - (m - 1)J
    assert pts[0] == [0.0, 3.0, 6.0, 9.0]
    assert pts[-1] == [10.0, 13.0, 16.0, 19.0]


def test_embedding_validation():
    s = [float(t) for t in range(20)]
    for call in (lambda: embed(s, 0, 1),
                 lambda: embed(s, 2, 0),
                 lambda: embed(s, 12, 3),
                 lambda: embed([1.0, 2.0], 2, 1)):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_delay_and_mean_period_on_a_cosine():
    period = 60
    c = [math.cos(2 * math.pi * t / period) for t in range(1200)]
    # rho(k) = cos(2 pi k / P), so the 1 - 1/e crossing is at
    # k = P acos(1 - 1/e) / (2 pi).
    want = P = period * math.acos(1.0 - 1.0 / math.e) / (2.0 * math.pi)
    assert abs(autocorrelation_lag(c) - want) <= 1.0
    assert abs(mean_period(c) - period) <= 1.0
    assert P > 0


def test_neighbours_respect_the_temporal_constraint():
    dv = divergence_curve(_logistic(300), m=2, tau=1, min_sep=5,
                          max_steps=3)
    nn = dv["neighbour"]
    assert all(abs(j - nn[j]) > 5 for j in range(len(nn)) if nn[j] >= 0)
    # d_j(0) is a true minimum under that constraint
    pts = dv["points"]
    j = 10
    best = min(math.sqrt(sum((pts[j][k] - pts[i][k]) ** 2
                             for k in range(2)))
               for i in range(len(pts)) if abs(i - j) > 5)
    assert abs(dv["d0"][j] - best) < 1e-12


def test_logistic_map_matches_the_papers_table_1():
    r = lyapunov_exponent(_logistic(1200), embedding=3, tau=1, min_sep=1)
    assert abs(r["estimate"] - 0.693) < 0.05
    assert r["r_squared"] > 0.99


def test_henon_map_matches_the_papers_table_1():
    r = lyapunov_exponent(_henon(1200), embedding=2, tau=1, min_sep=1)
    assert abs(r["estimate"] - 0.418) < 0.05


def test_a_periodic_signal_reports_no_chaos():
    sine = [math.sin(2 * math.pi * t / 50.0) for t in range(800)]
    assert abs(lyapunov_exponent(sine, embedding=3,
                                 tau=12)["estimate"]) < 0.02


def test_scaling_the_series_leaves_the_exponent_alone():
    hn = _henon(800)
    base = lyapunov_exponent(hn, embedding=2, tau=1, min_sep=1)
    scaled = lyapunov_exponent([1000.0 * v for v in hn], embedding=2,
                               tau=1, min_sep=1)
    assert abs(scaled["estimate"] - base["estimate"]) < 1e-9
    # eq. 12: a constant offset in ln d does not move a slope
    assert abs(scaled["log_divergence"][0] - base["log_divergence"][0] -
               math.log(1000.0)) < 1e-9


def test_dt_only_rescales_the_slope():
    lg = _logistic(600)
    a = lyapunov_exponent(lg, embedding=3, tau=1, min_sep=1)
    b = lyapunov_exponent(lg, embedding=3, tau=1, min_sep=1, dt=0.5)
    assert abs(b["estimate"] * 0.5 - a["estimate"]) < 1e-9


def test_all_three_routes_agree_and_are_selectable():
    hn = _henon(1200)
    r = lyapunov_exponent(hn, embedding=2, tau=1, min_sep=1)
    assert abs(r["rosenstein"] - r["sato"]) < 0.05
    assert abs(r["rosenstein"] - r["sato_k"]) < 0.05
    for m in ("rosenstein", "sato", "sato_k"):
        assert lyapunov_exponent(hn, embedding=2, tau=1, min_sep=1,
                                 method=m)["estimate"] == r[m]


def test_fitting_into_the_plateau_destroys_the_estimate():
    hn = _henon(1200)
    good = lyapunov_exponent(hn, embedding=2, tau=1, min_sep=1)["estimate"]
    wide = lyapunov_exponent(hn, embedding=2, tau=1, min_sep=1,
                             fit=(0, 60))["estimate"]
    assert wide < 0.6 * good


def test_validation():
    lg = _logistic(200)
    for call in (lambda: lyapunov_exponent([1.0] * 5),
                 lambda: lyapunov_exponent(lg, dt=0.0),
                 lambda: lyapunov_exponent(lg, min_sep=-1),
                 lambda: lyapunov_exponent(lg, fit=(0, 1)),
                 lambda: lyapunov_exponent(lg, method="wolf"),
                 lambda: lyapunov_exponent([1.0] * 200),
                 lambda: autocorrelation_lag([2.0] * 50)):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_alias_and_legacy_signature():
    hn = _henon(400)
    assert largest_lyapunov is lyapunov_exponent
    assert (lyapunov_exponent(hn, 2, 1)["estimate"] ==
            lyapunov_exponent(hn, embedding=2, tau=1)["estimate"])
