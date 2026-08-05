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
    beta = np.zeros(p)
    loglik = 0.0
    info = np.zeros((p, p))
    n_events = int(np.sum(event))
    if n_events == 0:
        raise ValueError("no events in the data")
    for it in range(max_iter):
        eta = np.clip(X @ beta + offs, -500.0, 500.0)
        w = np.exp(eta)
        U = np.zeros(p)
        info = np.zeros((p, p))
        loglik = 0.0
        for idx in groups.values():
            ts = [stop[i] for i in idx if event[i] == 1.0]
            for tk in sorted(set(ts)):
                D = [i for i in idx if stop[i] == tk and event[i] == 1.0]
                R = [i for i in idx if start[i] < tk <= stop[i]]
                S0 = float(np.sum(np.asarray([w[i] for i in R])))
                S1 = np.zeros(p)
                S2 = np.zeros((p, p))
                for i in R:
                    xi = X[i]
                    S1 = S1 + w[i] * xi
                    S2 = S2 + w[i] * np.outer(xi, xi)
                d = float(len(D))
                xbar = S1 / S0
                for i in D:
                    loglik += float(eta[i])
                    U = U + X[i]
                loglik -= d * float(np.log(S0))
                U = U - d * xbar
                info = info + d * (S2 / S0 - np.outer(xbar, xbar))
        step = np.linalg.solve(info, U)
        beta = beta + step
        if float(np.max(np.abs(step))) < tol:
            break
    cov = np.linalg.inv(info)
    diag = [float(cov[j, j]) for j in range(p)]
    if any(v <= 0.0 or v != v for v in diag) or float(np.max(np.abs(beta))) > 50.0:
        raise ValueError("partial likelihood is monotone or information singular")
    se = np.sqrt(np.asarray(diag))
    return {"beta": beta, "se": se, "cov": cov, "loglik": loglik,
            "n_iter": it + 1, "n_events": n_events}
