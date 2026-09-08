"""Tests for gwrcal (Fotheringham, Brunsdon & Charlton 2002)."""

import math

from morie.fn import _schab_gwr as G
from morie.fn.gwrcal import (bandwidth_profile, global_ols_aicc,
                             gwr_bandwidth_select, gwr_calibrate)


def _lcg(seed):
    st = [seed]

    def f():
        st[0] = (1103515245 * st[0] + 12345) % (1 << 31)
        return st[0] / float(1 << 31)
    return f


def _gauss(r):
    return math.sqrt(-2 * math.log(max(r(), 1e-12))) * \
        math.cos(2 * math.pi * r())


def _panel(seed=3, varying=True, n=60):
    r = _lcg(seed)
    y, X, C = [], [], []
    for _ in range(n):
        u, v = 10 * r(), 10 * r()
        x = _gauss(r)
        b = (0.5 + 0.4 * u) if varying else 2.0
        y.append(1.0 + b * x + 0.3 * _gauss(r))
        X.append([1.0, x])
        C.append([u, v])
    return y, X, C


def test_a_huge_bandwidth_is_ordinary_least_squares():
    from morie.fn import _array_core as np
    y, X, C = _panel()
    D = G.pairwise_distances(C)
    fit = G.gwr_fit(y, X, D, 1e6, "gaussian", False)
    beta, _, _, _ = np.linalg.lstsq(np.asarray(X, dtype=float),
                                    np.asarray(y, dtype=float))
    for i in (0, 17, len(y) - 1):
        for j in (0, 1):
            assert abs(float(fit["params"][i][j]) - float(beta[j])) < 1e-6
    assert abs(float(fit["tr_S"]) - 2.0) < 1e-6
    assert abs(G.aicc_from_parts(len(y), float(fit["sigma2"]),
                                 float(fit["tr_S"])) -
               global_ols_aicc(y, X)) < 1e-6


def test_the_criteria_match_their_formulae():
    y, X, C = _panel()
    D = G.pairwise_distances(C)
    fit = G.gwr_fit(y, X, D, 2.0, "gaussian", False)
    n = len(y)
    s2, tr = float(fit["sigma2"]), float(fit["tr_S"])
    want_aic = (2 * n * math.log(math.sqrt(s2)) +
                n * math.log(2 * math.pi) + n + tr)
    want_aicc = (2 * n * math.log(math.sqrt(s2)) +
                 n * math.log(2 * math.pi) +
                 n * (n + tr) / (n - 2.0 - tr))
    assert abs(G.aic_from_parts(n, s2, tr) - want_aic) < 1e-9
    assert abs(G.aicc_from_parts(n, s2, tr) - want_aicc) < 1e-9
    assert G.aicc_from_parts(n, s2, tr) > G.aic_from_parts(n, s2, tr)


def test_cv_is_leave_one_out():
    from morie.fn import _array_core as np
    y, X, C = _panel(n=30)
    D = G.pairwise_distances(C)
    cv = G.cv_score(y, X, D, 2.0, "gaussian", False)
    by_hand = 0.0
    for i in range(len(y)):
        w = [float(t) for t in G.kernel_weights(D[i], 2.0, "gaussian")]
        w[i] = 0.0
        b_i, _, _ = G._wls(np.asarray(X, dtype=float),
                           np.asarray(y, dtype=float),
                           np.asarray(w, dtype=float))
        by_hand += (y[i] - sum(X[i][j] * float(b_i[j])
                               for j in range(2))) ** 2
    assert abs(cv - by_hand) < 1e-8


def test_varying_coefficients_select_a_finite_bandwidth():
    res = gwr_calibrate(*_panel(varying=True))
    assert res["at_boundary"] is None
    assert res["aicc_improvement"] > 20.0
    assert res["tr_S"] > 2.0


def test_constant_coefficients_earn_nothing():
    res = gwr_calibrate(*_panel(varying=False))
    assert res["at_boundary"] == "upper"
    assert abs(res["tr_S"] - 2.0) < 0.6
    assert res["aicc_improvement"] < 3.0


def test_local_slopes_recover_a_planted_gradient():
    y, X, C = _panel(varying=True)
    res = gwr_calibrate(y, X, C)
    east = sorted(range(len(C)), key=lambda i: C[i][0])
    west_b = sum(res["coefficients"][i][1] for i in east[:15]) / 15.0
    east_b = sum(res["coefficients"][i][1] for i in east[-15:]) / 15.0
    assert west_b < east_b


def test_plain_aic_degenerates_where_aicc_does_not():
    y, X, C = _panel()
    aic = gwr_calibrate(y, X, C, criterion="aic")
    aicc = gwr_calibrate(y, X, C, criterion="aicc")
    assert aic["at_boundary"] == "lower"
    assert aicc["at_boundary"] is None
    assert aic["bandwidth"] < 0.1 * aicc["bandwidth"]


def test_all_kernels_and_criteria_run():
    y, X, C = _panel(n=40)
    for kern in ("gaussian", "bisquare", "tricube", "boxcar"):
        assert gwr_calibrate(y, X, C, kernel=kern)["bandwidth"] > 0
    for crit in ("aicc", "cv", "aic"):
        assert gwr_calibrate(y, X, C, criterion=crit)["criterion"] == crit


def test_adaptive_bandwidth_is_an_integer_neighbour_count():
    y, X, C = _panel(n=40)
    res = gwr_calibrate(y, X, C, adaptive=True)
    assert isinstance(res["bandwidth"], int)
    assert 3 <= res["bandwidth"] <= len(y)
    try:
        gwr_calibrate(y, X, C, adaptive=True, search="golden")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_profile_is_returned():
    y, X, C = _panel(n=30)
    grid, prof = bandwidth_profile(y, X, C, n_points=12)
    assert len(grid) == len(prof) == 12
    assert all(v == v for v in prof)


def test_validation():
    y, X, C = _panel(n=20)
    for call in (lambda: gwr_calibrate([1.0, 2.0], [[1.0], [1.0]],
                                       [[0.0], [1.0]]),
                 lambda: gwr_calibrate(y, X[:-1], C),
                 lambda: gwr_calibrate(y, X, C[:-1]),
                 lambda: gwr_calibrate(y, X, C, kernel="epanechnikov"),
                 lambda: gwr_calibrate(y, X, C, criterion="bic"),
                 lambda: gwr_calibrate(y, X, C, search="brent"),
                 lambda: gwr_calibrate(y, X, C, bounds=(5.0, 1.0)),
                 lambda: gwr_calibrate([float("nan")] + y[1:], X, C)):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_alias():
    y, X, C = _panel(n=25)
    a = gwr_bandwidth_select(y, X, C)
    b = gwr_calibrate(y, X, C)
    assert a["bandwidth"] == b["bandwidth"]
