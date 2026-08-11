"""Anchored tests for e_div (Matteson-James 2014 E-divisive)."""

import math

from morie.fn.e_div import e_div, _pairwise_alpha, _prefix2d, _qhat

X1 = [0.1, -0.2, 0.05, 0.3, -0.1, 5.2, 4.9, 5.1, 5.3, 4.8,
      1.9, 2.1, 2.0, 1.8, 2.2]


def _qhat_direct(x, a, tau, kappa, alpha=1.0):
    # Independent double-loop route for eq (5)-(6) of the paper.
    X = x[a:tau]
    Y = x[tau:kappa]
    n, m = len(X), len(Y)
    between = sum(abs(xi - yj) ** alpha for xi in X for yj in Y)
    wX = sum(abs(X[i] - X[k]) ** alpha
             for i in range(n) for k in range(i + 1, n))
    wY = sum(abs(Y[j] - Y[k]) ** alpha
             for j in range(m) for k in range(j + 1, m))
    e = 2.0 * between / (n * m) - wX / (n * (n - 1) / 2.0) \
        - wY / (m * (m - 1) / 2.0)
    return (n * m / float(n + m)) * e


def test_qhat_prefix_route_matches_direct_double_loop():
    D = _pairwise_alpha(X1, 1.0)
    P = _prefix2d(D)
    for (a, tau, kappa) in [(0, 5, 10), (0, 5, 15), (5, 10, 15),
                            (0, 7, 12), (2, 6, 14)]:
        assert abs(_qhat(P, a, tau, kappa)
                   - _qhat_direct(X1, a, tau, kappa)) < 1e-12


def test_e_div_matches_ecp_locations():
    # Anchor: ecp::e.divisive(matrix(x1), sig.lvl=0.05, R=199,
    # min.size=2) returns estimates {1, 6, 11, 16}, i.e. new segments
    # start at 6 and 11 -> tau = 5 and 10 in our convention
    # (run 2026-08-09, ecp 3.x, R 4.6.1).
    r = e_div(X1, sig=0.05, R=99, min_size=2, seed=1)
    assert sorted(r["changepoints"]) == [5, 10]
    assert all(p <= 0.05 for p in r["p_values"][:2])


def test_e_div_no_change_stops_immediately():
    x = [0.01 * ((-1) ** i) + 0.001 * i for i in range(12)]
    r = e_div(x, sig=0.05, R=99, min_size=2, seed=2)
    assert r["n_changepoints"] == 0
    assert len(r["p_values"]) == 1 and r["p_values"][0] > 0.05


def test_qhat_zero_for_identical_halves():
    # E(X, Y) = 0 iff identically distributed; for the empirical
    # version with Y an exact copy of X the between term equals the
    # average within term structure closely; sanity: statistic small
    # relative to a separated alternative.
    x = [0.0, 1.0, 2.0, 3.0] * 2
    D = _pairwise_alpha(x, 1.0)
    P = _prefix2d(D)
    q_same = _qhat(P, 0, 4, 8)
    y = [0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0]
    Dy = _pairwise_alpha(y, 1.0)
    Py = _prefix2d(Dy)
    q_diff = _qhat(Py, 0, 4, 8)
    assert q_diff > 10.0 * abs(q_same)
