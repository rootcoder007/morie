# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared counting-process Cox core for the recurrent-event shelf.

Newton-Raphson on the Breslow partial likelihood over (start, stop]
counting-process risk sets, optionally stratified. Used by agrec
(Andersen-Gill 1982), pwpgt (Prentice-Williams-Peterson 1981) and
wlwmm (Wei-Lin-Weissfeld 1989). Breslow ties throughout: the three
source papers all write the likelihood in Breslow form.
"""

from . import _array_core as np

__all__ = ["cox_counting_process"]


def _bi_clip(v, lo=-500.0, hi=500.0):
    return lo if v < lo else (hi if v > hi else v)


def cox_counting_process(start, stop, event, X, strata=None,
                         max_iter=50, tol=1e-9, offset=None):
    start = np.asarray(start, dtype=float)
    stop = np.asarray(stop, dtype=float)
    event = np.asarray(event, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape((-1, 1))
    if X.shape[0] != stop.shape[0]:
        X = X.T
    n = stop.shape[0]
    if start.shape[0] != n or event.shape[0] != n or X.shape[0] != n:
        raise ValueError("start, stop, event and X must have equal length")
    if np.any(stop <= start):
        raise ValueError("every interval needs stop > start")
    if not all(v in (0.0, 1.0) for v in event.tolist()):
        raise ValueError("event must be 0 or 1")
    p = X.shape[1]
    if strata is None:
        strata = [0] * n
    else:
        strata = list(strata)
        if len(strata) != n:
            raise ValueError("strata must match the number of rows")

    groups = {}
    for i, s in enumerate(strata):
        groups.setdefault(s, []).append(i)

    if offset is None:
        offs = np.zeros(n)
    else:
        offs = np.asarray(offset, dtype=float)
        if offs.shape[0] != n:
            raise ValueError("offset must match the number of rows")
    # the Newton loop runs on plain floats: building small arrays per
    # risk-set member made one fit cost ~0.7 s at n = 24, and the shared
    # frailty EM calls this fit dozens of times per theta
    import math as _m
    Xl = [[float(v) for v in row] for row in X.tolist()]
    st = [float(v) for v in start.tolist()]
    sp = [float(v) for v in stop.tolist()]
    ev = [float(v) for v in event.tolist()]
    of = [float(v) for v in offs.tolist()]
    n_events = int(sum(ev))
    if n_events == 0:
        raise ValueError("no events in the data")
    plan = []
    for idx in groups.values():
        ts = sorted(set(sp[i] for i in idx if ev[i] == 1.0))
        for tk in ts:
            D = [i for i in idx if sp[i] == tk and ev[i] == 1.0]
            R = [i for i in idx if st[i] < tk <= sp[i]]
            plan.append((D, R))
    b_ = [0.0] * p
    loglik = 0.0
    info_l = [[0.0] * p for _ in range(p)]
    it = 0
    for it in range(max_iter):
        eta = [_bi_clip(sum(Xl[i][j] * b_[j] for j in range(p)) + of[i])
               for i in range(n)]
        w = [_m.exp(v) for v in eta]
        U = [0.0] * p
        info_l = [[0.0] * p for _ in range(p)]
        loglik = 0.0
        for D, R in plan:
            S0 = sum(w[i] for i in R)
            S1 = [0.0] * p
            S2 = [[0.0] * p for _ in range(p)]
            for i in R:
                xi = Xl[i]
                wi = w[i]
                for a_ in range(p):
                    S1[a_] = S1[a_] + wi * xi[a_]
                    for c_ in range(p):
                        S2[a_][c_] = S2[a_][c_] + wi * (xi[a_] * xi[c_])
            d = float(len(D))
            xbar = [v / S0 for v in S1]
            for i in D:
                loglik += eta[i]
                for a_ in range(p):
                    U[a_] = U[a_] + Xl[i][a_]
            loglik -= d * _m.log(S0)
            for a_ in range(p):
                U[a_] = U[a_] - d * xbar[a_]
                for c_ in range(p):
                    info_l[a_][c_] = info_l[a_][c_] + d * (S2[a_][c_] / S0
                                                           - xbar[a_] * xbar[c_])
        try:
            step = list(np.linalg.solve(np.array(info_l), np.array(U))._flat())
        except Exception:
            raise ValueError("partial likelihood is monotone or information singular")
        b_ = [b_[j] + step[j] for j in range(p)]
        if max(abs(v) for v in step) < tol:
            break
    beta = np.array(b_)
    info = np.array(info_l)
    cov = np.linalg.inv(info)
    diag = [float(cov[j, j]) for j in range(p)]
    if any(v <= 0.0 or v != v for v in diag) or float(np.max(np.abs(beta))) > 50.0:
        raise ValueError("partial likelihood is monotone or information singular")
    se = np.sqrt(np.asarray(diag))
    return {"beta": beta, "se": se, "cov": cov, "loglik": loglik,
            "n_iter": it + 1, "n_events": n_events}
