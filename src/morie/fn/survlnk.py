# SPDX-License-Identifier: AGPL-3.0-or-later
"""Link-function survival regression."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["survlnk", "link_function_survival"]


def _links(link):
    if link == "cloglog":
        # h = 1 - exp(-exp(eta)); grouped proportional hazards
        def inv(eta):
            return 1.0 - np.exp(-np.exp(np.clip(eta, -30.0, 30.0)))
    elif link == "logit":
        def inv(eta):
            return 1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0)))
    else:
        raise ValueError("link must be 'cloglog' or 'logit'")
    return inv


def link_function_survival(time, event, X, link="cloglog",
                           max_iter=100, tol=1e-10):
    """
    Discrete-time survival regression with a chosen hazard link.

    The observed event times define discrete risk intervals; subject i
    at risk at event time t_k contributes a Bernoulli observation with
    hazard h_ik = g^{-1}(alpha_k + beta' x_i). The complementary
    log-log link makes this the grouped proportional-hazards model of
    Prentice-Gloeckler; the logit link gives Cox's discrete
    proportional-odds model. Klein & Moeschberger (2003), Survival
    Analysis, 2nd ed., Section 8.4 treat both.

    Fitted by full-likelihood Newton-Raphson over (alpha_1..alpha_K,
    beta) via finite scoring on the expanded person-period data.

    Returns
    -------
    result : RichResult
        Keys: estimate (beta), se, alpha (per-interval intercepts),
        event_times, loglik, n_iter, link.
    """
    time = np.asarray(time, dtype=float)
    ev = np.asarray(event, dtype=float)
    Xa = np.asarray(X, dtype=float)
    if Xa.ndim == 1:
        Xa = Xa.reshape((-1, 1))
    if Xa.shape[0] != time.shape[0]:
        Xa = Xa.T
    n, p = Xa.shape
    inv = _links(link)
    etimes = sorted(set(float(time[i]) for i in range(n) if ev[i] == 1.0))
    K = len(etimes)
    if K == 0:
        raise ValueError("no events in the data")
    # person-period expansion
    rows = []   # (k, i, y)
    for i in range(n):
        for k, tk in enumerate(etimes):
            if time[i] >= tk:
                y = 1.0 if (time[i] == tk and ev[i] == 1.0) else 0.0
                rows.append((k, i, y))
            else:
                break
    q = K + p
    theta = np.zeros(q)
    ll = 0.0
    info = np.zeros((q, q))
    for it in range(max_iter):
        U = np.zeros(q)
        info = np.zeros((q, q))
        ll = 0.0
        for (k, i, y) in rows:
            eta = float(theta[k]) + float(Xa[i] @ theta[K:])
            h = float(inv(np.asarray([eta]))[0])
            h = min(max(h, 1e-12), 1.0 - 1e-12)
            # d loglik / d eta for Bernoulli(y; h(eta))
            if link == "cloglog":
                ee = np.exp(np.clip(np.asarray([eta]), -30.0, 30.0))[0]
                dh = float(ee * (1.0 - h))
            else:
                dh = h * (1.0 - h)
            ll += y * np.log(h) + (1.0 - y) * np.log(1.0 - h)
            gscal = (y - h) / (h * (1.0 - h)) * dh
            wscal = dh * dh / (h * (1.0 - h))
            g = np.zeros(q)
            g[k] = 1.0
            for j in range(p):
                g[K + j] = float(Xa[i, j])
            U = U + gscal * g
            info = info + wscal * np.outer(g, g)
        step = np.linalg.solve(info, U)
        theta = theta + step
        if float(np.max(np.abs(step))) < tol:
            break
    cov = np.linalg.inv(info)
    dg = [float(cov[K + j, K + j]) for j in range(p)]
    if any(v <= 0.0 or v != v for v in dg):
        raise ValueError("information matrix is singular")
    return RichResult(payload={
        "estimate": theta[K:],
        "se": np.sqrt(np.asarray(dg)),
        "alpha": theta[:K],
        "event_times": np.asarray(etimes),
        "loglik": ll,
        "n_iter": it + 1,
        "link": link,
        "method": "Klein-Moeschberger (2003) sec. 8.4 discrete-time hazard regression",
    })


survlnk = link_function_survival


def cheatsheet():
    return "survlnk(time, event, X, link) -> discrete-time hazard regression (cloglog = grouped PH, logit = proportional odds)."
