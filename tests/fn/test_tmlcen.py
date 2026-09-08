"""Tests for tmlcen -- causal effect under censoring by IPCW.

Replaces a generated test that called a stub returning mean(y). Full
anchor: ledger/wave3/anchor_tmlcen.py.
"""

import math

import pytest

from morie.fn import _array_core as np
from morie.fn.tmlcen import (censoring_survival, coarsen_interval,
                             ipcw_interval, tmle_censoring)

N = 2000
K = 6


def expit(z):
    return 1.0 / (1.0 + math.exp(-z))


def death_hazard(a, w):
    return expit(-2.0 - 0.8 * a + 0.7 * w)


@pytest.fixture(scope="module")
def censored():
    """Censoring tracks the same covariate that drives death, so
    ignoring it biases the survival curves."""
    rng = np.random.default_rng(5)
    W = [rng.standard_normal() for _ in range(N)]
    A = [1.0 if rng.uniform() < expit(0.3 * W[i]) else 0.0
         for i in range(N)]
    T, C = [], []
    for i in range(N):
        hz = death_hazard(A[i], W[i])
        t = K
        for kk in range(K):
            if rng.uniform() < hz:
                t = kk
                break
        T.append(t)
        ch = expit(-1.2 + 2.0 * W[i])
        cc = K
        for kk in range(K):
            if rng.uniform() < ch:
                cc = kk
                break
        C.append(cc)
    truth = (sum((1.0 - death_hazard(1.0, W[i])) ** K for i in range(N))
             - sum((1.0 - death_hazard(0.0, W[i])) ** K
                   for i in range(N))) / N
    return {"W": [[W[i]] for i in range(N)], "A": A,
            "obs": [min(T[i], C[i]) for i in range(N)],
            "ev": [1.0 if (T[i] <= C[i] and T[i] < K) else 0.0
                   for i in range(N)],
            "cen": [1.0 if (C[i] < T[i] and C[i] < K) else 0.0
                    for i in range(N)],
            "truth": truth}


def test_coarsening_matches_section_8_5():
    """L is the last Delta=0, R the first Delta=1, with both boundary
    cases spelled out in the source."""
    assert coarsen_interval([1, 2, 3, 4], [0, 0, 1, 1]) == (2.0, 3.0)
    assert coarsen_interval([1, 2, 3], [1, 1, 1]) == (0.0, 1.0)
    assert coarsen_interval([1, 2, 3], [0, 0, 0]) == (3.0, float("inf"))
    # unsorted input is sorted first
    assert coarsen_interval([3, 1, 2], [1, 0, 0]) == (2.0, 3.0)


def test_coarsening_validates():
    with pytest.raises(ValueError):
        coarsen_interval([1, 2], [0])
    with pytest.raises(ValueError):
        coarsen_interval([1], [2])
    with pytest.raises(ValueError):
        coarsen_interval([], [])


def test_ipcw_recovers_the_survival_difference(censored):
    d = censored
    r = tmle_censoring(d["obs"], d["ev"], d["cen"], d["A"], d["W"])
    assert r["estimate"] == pytest.approx(d["truth"], abs=0.06)


def test_the_weights_reduce_bias_not_just_one_draw():
    """The comparator has to be the right one.

    Against the same hazard model fitted WITHOUT weights, IPCW wins on
    roughly half of draws and loses on the rest -- because censoring
    here depends only on W and that model already conditions on W, so
    conditioning and weighting are two solutions to one problem and
    neither dominates. That is a fact about the design.

    Against a hazard model that DROPS the covariates, which is what
    ignoring informative censoring actually looks like, the weights
    remove real bias. Averaged over independent draws so the claim is
    about bias rather than one sample."""
    ipcw_err = naive_err = 0.0
    seeds = (11, 12, 13, 14, 15)
    for sd in seeds:
        rng = np.random.default_rng(sd)
        n = 1500
        W = [rng.standard_normal() for _ in range(n)]
        A = [1.0 if rng.uniform() < expit(0.3 * W[i]) else 0.0
             for i in range(n)]
        T, C = [], []
        for i in range(n):
            hz = death_hazard(A[i], W[i])
            t = K
            for kk in range(K):
                if rng.uniform() < hz:
                    t = kk
                    break
            T.append(t)
            ch = expit(-1.2 + 2.0 * W[i])
            cc = K
            for kk in range(K):
                if rng.uniform() < ch:
                    cc = kk
                    break
            C.append(cc)
        truth = (sum((1.0 - death_hazard(1.0, W[i])) ** K
                     for i in range(n))
                 - sum((1.0 - death_hazard(0.0, W[i])) ** K
                       for i in range(n))) / n
        r = tmle_censoring([min(T[i], C[i]) for i in range(n)],
                           [1.0 if (T[i] <= C[i] and T[i] < K) else 0.0
                            for i in range(n)],
                           [1.0 if (C[i] < T[i] and C[i] < K) else 0.0
                            for i in range(n)],
                           A, [[W[i]] for i in range(n)])
        ipcw_err += abs(r["estimate"] - truth)
        naive_err += abs(r["unadjusted"] - truth)
    assert ipcw_err / len(seeds) < naive_err / len(seeds)


def test_survival_curves_are_curves(censored):
    d = censored
    r = tmle_censoring(d["obs"], d["ev"], d["cen"], d["A"], d["W"])
    for key in ("survival_treated", "survival_control"):
        s = r[key]
        assert all(s[i + 1] <= s[i] + 1e-12 for i in range(len(s) - 1))
        assert all(0.0 <= v <= 1.0 for v in s)
    # treatment lowers the hazard here, so its curve sits above control
    assert all(r["survival_treated"][i] >= r["survival_control"][i]
               for i in range(len(r["grid"])))


def test_censoring_survival_is_a_survival_curve(censored):
    d = censored
    G, grid, _ = censoring_survival(d["obs"], d["cen"], A=d["A"],
                                    W=d["W"])
    assert all(0.0 < v <= 1.0 for row in G for v in row)
    assert all(row[i + 1] <= row[i] + 1e-12
               for row in G for i in range(len(row) - 1))


def test_interval_ipcw_and_its_positivity_guards(censored):
    d = censored
    rng = np.random.default_rng(3)
    M = 4
    mon = [sorted(1.0 + 4.0 * float(rng.uniform()) for _ in range(M))
           for _ in range(N)]
    dele = [[1.0 if float(rng.uniform()) < 0.4 else 0.0
             for _ in range(M)] for _ in range(N)]
    g = [0.5] * N
    gc = [[0.25] * M for _ in range(N)]
    psi = ipcw_interval(d["W"], d["A"], mon, dele, a=1.0, g=g, gc=gc)
    assert psi > 0.0
    with pytest.raises(ValueError):
        ipcw_interval(d["W"], d["A"], mon, dele, a=1.0, g=[0.0] * N,
                      gc=gc)
    with pytest.raises(ValueError):
        ipcw_interval(d["W"], d["A"], mon, dele, a=1.0, g=g,
                      gc=[[0.0] * M for _ in range(N)])


def test_argument_checks(censored):
    d = censored
    with pytest.raises(ValueError):
        tmle_censoring([1.0], [1.0], [1.0], [1.0], [[0.0]])
    with pytest.raises(ValueError):
        tmle_censoring(d["obs"], d["ev"], d["cen"], d["A"], d["W"],
                       kind="nope")
