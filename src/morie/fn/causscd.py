r"""Synthetic difference in differences.

Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., & Wager, S.
(2021) "Synthetic Difference-in-Differences", *American Economic Review*
111(12), 4088-4118.

DID assumes the controls are a good comparison on average; synthetic
control reweights them to match the treated unit's pre-period path but
drops the two-way structure. SDID keeps both: it solves the same
weighted two-way regression as DID, but with unit weights
:math:`\hat\omega_i` and time weights :math:`\hat\lambda_t` chosen so that
the reweighted controls track the treated units before treatment.

.. math::

   (\hat\tau, \hat\mu, \hat\alpha, \hat\beta) =
   \arg\min \sum_{i,t}
     \bigl(Y_{it} - \mu - \alpha_i - \beta_t - W_{it}\tau\bigr)^2
     \hat\omega_i \hat\lambda_t

The unit weights solve, over the simplex,

.. math::

   \min_{\omega_0, \omega}\ \sum_{t \le T_{pre}}
     \Bigl(\omega_0 + \sum_{i \in co} \omega_i Y_{it}
           - \frac{1}{N_{tr}}\sum_{i \in tr} Y_{it}\Bigr)^2
     + \zeta^2 T_{pre} \lVert \omega \rVert^2 ,

with the intercept :math:`\omega_0` free -- which is what lets SDID match
a *parallel* control path rather than an identical one, the difference
from synthetic control. Time weights are the same problem transposed,
without the penalty.

Every estimator in the family is the same weighted average of adjusted
outcomes (their eq. 2.4), so ``method="did"``, ``"sc"`` and ``"sdid"``
differ only in the weights: DID uses :math:`1/N_{co}` and uniform time
weights, SC uses fitted unit weights and uniform time weights, SDID uses
both.
"""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["causscd", "sdid", "unit_weights", "time_weights"]


def _grid(Y, treated, t_post):
    rows = [[float(v) for v in r] for r in np.asarray(Y, dtype=float)]
    n = len(rows)
    if n < 2:
        raise ValueError("causscd: need at least two units")
    T = len(rows[0])
    if any(len(r) != T for r in rows):
        raise ValueError("causscd: Y is ragged")
    for r in rows:
        for v in r:
            if v != v or v in (float("inf"), float("-inf")):
                raise ValueError("causscd: Y contains a non-finite value")
    tr = [bool(v) for v in treated]
    if len(tr) != n:
        raise ValueError("causscd: treated must have one flag per unit")
    t_post = int(t_post)
    if not 1 <= t_post < T:
        raise ValueError("causscd: t_post must lie in 1..T-1 (it is the "
                         "number of pre-treatment periods)")
    if not any(tr):
        raise ValueError("causscd: no treated units")
    if all(tr):
        raise ValueError("causscd: no control units")
    return rows, n, T, tr, t_post


def _simplex_fit(cols, target, penalty, iters=2000, tol=1e-12):
    """Minimise ||intercept + sum_k w_k cols_k - target||^2 + penalty
    ||w||^2 over the simplex, by projected gradient with a free
    intercept."""
    m = len(cols)
    L = len(target)
    w = [1.0 / m] * m
    step = None
    for _ in range(int(iters)):
        fit = [sum(w[k] * cols[k][t] for k in range(m)) for t in range(L)]
        icept = sum(target[t] - fit[t] for t in range(L)) / L
        resid = [icept + fit[t] - target[t] for t in range(L)]
        grad = [2.0 * sum(resid[t] * cols[k][t] for t in range(L)) +
                2.0 * penalty * w[k] for k in range(m)]
        if step is None:
            gnorm = math.sqrt(sum(g * g for g in grad)) or 1.0
            step = 1.0 / gnorm
        cand = [w[k] - step * grad[k] for k in range(m)]
        cand = _project_simplex(cand)
        if max(abs(cand[k] - w[k]) for k in range(m)) < tol:
            w = cand
            break
        w = cand
    fit = [sum(w[k] * cols[k][t] for k in range(m)) for t in range(L)]
    icept = sum(target[t] - fit[t] for t in range(L)) / L
    return w, icept


def _project_simplex(v):
    """Euclidean projection onto {w >= 0, sum w = 1}."""
    m = len(v)
    u = sorted(v, reverse=True)
    css = 0.0
    rho, theta = 0, 0.0
    for k in range(m):
        css += u[k]
        t = (css - 1.0) / (k + 1)
        if u[k] - t > 0:
            rho, theta = k + 1, t
    return [max(0.0, x - theta) for x in v]


def unit_weights(Y, treated, t_post, zeta=None):
    r"""The :math:`\hat\omega` of the paper's eq. 2.8, over the simplex,
    with a free intercept and the ridge penalty :math:`\zeta`."""
    rows, n, T, tr, t_post = _grid(Y, treated, t_post)
    co = [i for i in range(n) if not tr[i]]
    trt = [i for i in range(n) if tr[i]]
    pre = list(range(t_post))
    target = [sum(rows[i][t] for i in trt) / len(trt) for t in pre]
    cols = [[rows[i][t] for t in pre] for i in co]
    if zeta is None:
        diffs = []
        for i in co:
            for t in range(1, t_post):
                diffs.append(rows[i][t] - rows[i][t - 1])
        if len(diffs) > 1:
            mu = sum(diffs) / len(diffs)
            sd = math.sqrt(sum((v - mu) ** 2 for v in diffs) /
                           (len(diffs) - 1))
        else:
            sd = 1.0
        zeta = (len(trt) * (T - t_post)) ** 0.25 * sd
    w, icept = _simplex_fit(cols, target, (zeta ** 2) * t_post)
    full = [0.0] * n
    for k, i in enumerate(co):
        full[i] = w[k]
    return full, icept, zeta


def time_weights(Y, treated, t_post):
    r"""The :math:`\hat\lambda` of eq. 2.9: the same fit transposed, over
    the pre-periods, with no penalty."""
    rows, n, T, tr, t_post = _grid(Y, treated, t_post)
    co = [i for i in range(n) if not tr[i]]
    post = list(range(t_post, T))
    target = [sum(rows[i][t] for t in post) / len(post) for i in co]
    cols = [[rows[i][t] for i in co] for t in range(t_post)]
    w, icept = _simplex_fit(cols, target, 0.0)
    full = [0.0] * T
    for t in range(t_post):
        full[t] = w[t]
    return full, icept


def sdid(Y, treated, t_post, method="sdid", zeta=None):
    """The estimator of eq. 2.4-2.5 for any of the three weightings."""
    rows, n, T, tr, t_post = _grid(Y, treated, t_post)
    if method not in ("sdid", "did", "sc"):
        raise ValueError("causscd: method must be 'sdid', 'did' or 'sc'")
    co = [i for i in range(n) if not tr[i]]
    trt = [i for i in range(n) if tr[i]]
    pre, post = list(range(t_post)), list(range(t_post, T))

    if method == "did":
        om = [0.0] * n
        for i in co:
            om[i] = 1.0 / len(co)
        lam = [0.0] * T
        for t in pre:
            lam[t] = 1.0 / len(pre)
        zeta_used = 0.0
    else:
        om, _, zeta_used = unit_weights(Y, treated, t_post, zeta)
        if method == "sc":
            lam = [0.0] * T
            for t in pre:
                lam[t] = 1.0 / len(pre)
        else:
            lam, _ = time_weights(Y, treated, t_post)

    def wavg_pre(i):
        return sum(lam[t] * rows[i][t] for t in pre)

    def avg_post(i):
        return sum(rows[i][t] for t in post) / len(post)

    delta = dict((i, avg_post(i) - wavg_pre(i)) for i in range(n))
    d_tr = sum(delta[i] for i in trt) / len(trt)
    d_co = sum(om[i] * delta[i] for i in co)
    tau = d_tr - d_co
    return RichResult(payload={
        "estimate": tau,
        "tau": tau,
        "unit_weights": om,
        "time_weights": lam,
        "zeta": zeta_used,
        "delta_treated": d_tr,
        "delta_control": d_co,
        "method_name": method,
        "n_treated": len(trt),
        "n_control": len(co),
        "t_pre": t_post,
        "t_post": T - t_post,
        "method": ("synthetic DID (Arkhangelsky, Athey, Hirshberg, "
                   "Imbens & Wager 2021), weighting '%s'" % method),
        "note": ("all three weightings are the same estimator of eq. "
                 "2.4; DID uses 1/N_co and uniform time weights, SC "
                 "fitted unit weights only, SDID both"),
    })


def causscd(Y, treated, t_post, zeta=None):
    """SDID with DID and SC reported alongside for comparison."""
    out = sdid(Y, treated, t_post, "sdid", zeta)
    p = dict(out.payload)
    p["did"] = sdid(Y, treated, t_post, "did")["tau"]
    p["sc"] = sdid(Y, treated, t_post, "sc", zeta)["tau"]
    p["sdid"] = out["tau"]
    return RichResult(payload=p)


def cheatsheet():
    return ("causscd: synthetic DID (Arkhangelsky et al. 2021). Same "
            "weighted two-way regression as DID, but with unit weights "
            "fitted over the simplex WITH a free intercept (so the "
            "controls need only be parallel to the treated path, not "
            "identical to it) and time weights fitted the same way "
            "transposed. method='did' uses 1/N_co and uniform time "
            "weights; 'sc' uses unit weights only; 'sdid' uses both.")
