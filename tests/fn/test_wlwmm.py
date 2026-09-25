"""Tests for wlwmm.wlw_marginal_model (Wei, Lin & Weissfeld 1989)."""

import math

import pytest

from morie.fn.wlwmm import wlw_marginal_model, wlwmm

N_SUBJ = 30
OCCURRENCES = (1, 2)


def _data():
    """Two occurrences per subject on the TOTAL time scale, with ties
    and censoring, and a binary plus a continuous covariate."""
    time, event, X, occ = [], [], [], []
    for s in range(N_SUBJ):
        treat = float(s % 2)
        # age must not order the event times, or the partial likelihood is
        # monotone in its coefficient and no MLE exists
        age = 0.5 * (((3 * s + 1) % 7) - 3)
        for k in OCCURRENCES:
            # first occurrence earlier than the second; ties every 5 subjects
            base = 2.0 + k * 3.0 + (s % 5)
            time.append(base - 0.75 * treat)
            event.append(0.0 if s % 7 == 0 else 1.0)
            X.append([treat, age])
            occ.append(k)
    return time, event, X, occ


def _breslow(beta, time, event, X, occ):
    """Stratified Breslow partial log-likelihood, score and information,
    written out here so the fit is checked against arithmetic."""
    p = len(beta)
    eta = [sum(b * v for b, v in zip(beta, row)) for row in X]
    w = [math.exp(e) for e in eta]
    ll = 0.0
    U = [0.0] * p
    info = [[0.0] * p for _ in range(p)]
    for k in sorted(set(occ)):
        idx = [i for i, o in enumerate(occ) if o == k]
        for tk in sorted({time[i] for i in idx if event[i] == 1.0}):
            D = [i for i in idx if time[i] == tk and event[i] == 1.0]
            R = [i for i in idx if tk <= time[i]]      # start = 0 for all
            S0 = sum(w[i] for i in R)
            S1 = [sum(w[i] * X[i][j] for i in R) for j in range(p)]
            S2 = [[sum(w[i] * X[i][j] * X[i][m] for i in R)
                   for m in range(p)] for j in range(p)]
            d = float(len(D))
            xbar = [S1[j] / S0 for j in range(p)]
            for i in D:
                ll += eta[i]
                for j in range(p):
                    U[j] += X[i][j]
            ll -= d * math.log(S0)
            for j in range(p):
                U[j] -= d * xbar[j]
                for m in range(p):
                    info[j][m] += d * (S2[j][m] / S0 - xbar[j] * xbar[m])
    return ll, U, info


def _inv2(m):
    det = m[0][0] * m[1][1] - m[0][1] * m[1][0]
    return [[m[1][1] / det, -m[0][1] / det], [-m[1][0] / det, m[0][0] / det]]


def test_wlwmm_basic():
    """The common beta solves the stratified Breslow score equation, and
    se is the inverse-information standard error at that solution."""
    time, event, X, occ = _data()
    res = wlw_marginal_model(time, event, X, occ)

    beta = [float(v) for v in res["estimate"]]
    assert len(beta) == 2
    assert all(math.isfinite(v) for v in beta)
    assert res["n_events"] == int(sum(event))

    ll, U, info = _breslow(beta, time, event, X, occ)
    # the score vanishes at the reported estimate
    for j in range(2):
        assert U[j] == pytest.approx(0.0, abs=1e-7)
    assert res["loglik"] == pytest.approx(ll, rel=1e-9, abs=1e-9)

    V = _inv2(info)
    se = [float(v) for v in res["se"]]
    for j in range(2):
        assert se[j] == pytest.approx(math.sqrt(V[j][j]), rel=1e-7)
        assert se[j] > 0.0

    # the score at any nearby point is not zero, so the solution is sharp
    off = _breslow([beta[0] + 0.3, beta[1]], time, event, X, occ)[1]
    assert abs(off[0]) > 1e-4
    # and the partial likelihood is maximised there
    for h in (0.05, 0.3):
        for j in range(2):
            up = list(beta)
            up[j] += h
            dn = list(beta)
            dn[j] -= h
            assert _breslow(up, time, event, X, occ)[0] < ll
            assert _breslow(dn, time, event, X, occ)[0] < ll

    # the treatment column delays the event, so its coefficient is positive
    # (later failure times under the Breslow risk sets used above)
    assert beta[0] > 0.0

    # one marginal fit per occurrence, each a 2-vector
    per = res["per_event_beta"]
    assert set(per) == set(OCCURRENCES)
    for k in OCCURRENCES:
        assert len([float(v) for v in per[k]]) == 2

    # wlwmm is the public alias
    assert [float(v) for v in wlwmm(time, event, X, occ)["estimate"]] == (
        pytest.approx(beta, rel=1e-9, abs=1e-12))


def test_wlwmm_is_invariant_to_shifting_and_scaling_the_covariates():
    """The partial likelihood depends on covariate differences only."""
    time, event, X, occ = _data()
    base = [float(v) for v in wlw_marginal_model(time, event, X, occ)["estimate"]]

    shifted = [[row[0] + 10.0, row[1] - 3.0] for row in X]
    s = [float(v) for v in
         wlw_marginal_model(time, event, shifted, occ)["estimate"]]
    assert s == pytest.approx(base, rel=1e-6, abs=1e-8)

    scaled = [[2.0 * row[0], row[1]] for row in X]
    z = [float(v) for v in
         wlw_marginal_model(time, event, scaled, occ)["estimate"]]
    assert z[0] == pytest.approx(base[0] / 2.0, rel=1e-6)
    assert z[1] == pytest.approx(base[1], rel=1e-6)


def test_wlwmm_edge():
    """One covariate, one occurrence, and the documented rejections."""
    time, event, X, occ = _data()
    x1 = [[row[0]] for row in X]
    res = wlw_marginal_model(time, event, x1, occ)
    b = [float(v) for v in res["estimate"]]
    assert len(b) == 1
    ll, U, info = _breslow(b, time, event, x1, occ)
    assert U[0] == pytest.approx(0.0, abs=1e-7)
    assert res["loglik"] == pytest.approx(ll, rel=1e-9, abs=1e-9)
    assert float(res["se"][0]) == pytest.approx(1.0 / math.sqrt(info[0][0]),
                                                rel=1e-7)

    # a single occurrence reduces to one marginal model, and the
    # stratified summary is that same model
    one = [i for i, o in enumerate(occ) if o == 1]
    t1 = [time[i] for i in one]
    e1 = [event[i] for i in one]
    m1 = [x1[i] for i in one]
    o1 = [1] * len(one)
    solo = wlw_marginal_model(t1, e1, m1, o1)
    assert set(solo["per_event_beta"]) == {1}
    assert float(solo["per_event_beta"][1][0]) == pytest.approx(
        float(solo["estimate"][0]), rel=1e-6, abs=1e-9)

    # an occurrence with no events contributes no marginal fit
    e_mixed = [0.0 if o == 2 else ev for ev, o in zip(event, occ)]
    mixed = wlw_marginal_model(time, e_mixed, x1, occ)
    assert set(mixed["per_event_beta"]) == {1}
    assert mixed["n_events"] == int(sum(e_mixed))

    with pytest.raises(ValueError):
        wlw_marginal_model([0.0] + time[1:], event, x1, occ)   # stop <= start
    with pytest.raises(ValueError):
        wlw_marginal_model(time, [0.5] * len(time), x1, occ)   # event not 0/1
    with pytest.raises(ValueError):
        wlw_marginal_model(time, [0.0] * len(time), x1, occ)   # no events
    with pytest.raises(ValueError):
        wlw_marginal_model(time, event, x1, occ[:-1])          # strata length
