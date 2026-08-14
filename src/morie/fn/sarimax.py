# morie.fn -- function file (rootcoder007/morie)
r"""Regression with seasonal ARIMA errors, and automatic order selection.

**The model.** A mean function in observed regressors, with the
disturbance following a multiplicative seasonal ARIMA:

.. math:: y_t = \boldsymbol\beta' \mathbf{x}_t + n_t, \qquad
          \phi_p(B)\Phi_P(B^s)\nabla^d\nabla_s^D n_t
          = \theta_q(B)\Theta_Q(B^s) a_t.

Fitting :math:`\boldsymbol\beta` by ordinary least squares and *then*
modelling the residuals is not the same thing and is not done here: the
regression and the error model are estimated together.

**How the regression is concentrated out.** The Kalman filter is linear
in the observation, so the innovations of :math:`y - X\beta` are
:math:`v_y - v_X\beta` with the *same* innovation variances
:math:`f_t`. Given the ARIMA parameters, the profile estimate is
therefore the exact generalised least squares solution

.. math:: \hat{\boldsymbol\beta} = \Big(\sum_t
          \frac{v_{x,t} v_{x,t}'}{f_t}\Big)^{-1}
          \sum_t \frac{v_{x,t} v_{y,t}}{f_t},

computed by running the same filter down each column of :math:`X`.
Only the ARIMA parameters are left to the optimiser, so a model with
ten regressors costs the optimiser nothing extra.

**Automatic order selection.** ``auto_order`` is the step-wise search
of Hyndman & Khandakar (2008) Sec. 3.2, in full:

*Step 1* fits four starting models -- :math:`(2,d,2)(1,D,1)`,
:math:`(0,d,0)(0,D,0)`, :math:`(1,d,0)(1,D,0)` and
:math:`(0,d,1)(0,D,1)` for seasonal data, and their non-seasonal
counterparts otherwise -- and keeps the one with the smallest AIC.

*Step 2* considers up to thirteen variations on the current model: one
of :math:`p, q, P, Q` moved by :math:`\pm 1`; :math:`p` and :math:`q`
both moved by :math:`\pm 1`; :math:`P` and :math:`Q` both moved by
:math:`\pm 1`; and the constant switched in or out. Whenever a lower
AIC turns up, that model becomes current and the step repeats. It
stops when no neighbour improves on the current model.

The paper's four constraints are enforced and reported, not assumed:
:math:`p, q \le 5`, :math:`P, Q \le 2`, any model whose AR or MA
polynomial has a root smaller than 1.001 in modulus is rejected, and
any model whose optimisation errors is rejected -- "the rationale here
is that any model that is difficult to fit is probably not a good model
for the data".

**What is deliberately not here.** Hyndman & Khandakar choose :math:`D`
by an extended Canova-Hansen test and :math:`d` by successive KPSS
tests. Neither test is implemented here, so ``auto_order`` takes
:math:`d` and :math:`D` as given rather than guessing them from
critical values I would have to invent; ``differencing_note`` says so
in the result. The constant is admitted only when :math:`d + D < 2`,
which is the paper's rule.

References
----------
Hyndman, R. J. & Khandakar, Y. (2008) "Automatic Time Series
Forecasting: The forecast Package for R", *Journal of Statistical
Software* 27(3), 1-22, doi:10.18637/jss.v027.i03. Sec. 3.1 for the
seasonal ARIMA definition and the AIC
:math:`-2\log L + 2(p+q+P+Q+k)` with :math:`k=1` when a constant is
included, for the statement that the likelihood is that of the
differenced data so AIC values across differencing orders are not
comparable, and for the rule admitting a constant only when
:math:`d+D<2`; Sec. 3.2 for the four starting models, the thirteen
neighbours, the stopping rule, and the four constraints including the
1.001 root threshold.

Box, G. E. P., Jenkins, G. M., Reinsel, G. C. & Ljung, G. M. (2016)
*Time Series Analysis: Forecasting and Control*, 5th edn, Wiley,
ISBN 978-1-118-67502-1, Sec. 9.5, for regression models with
autocorrelated (seasonal ARIMA) errors.
"""

import math

from . import _array_core as np
from . import sarima as _sa
from ._richresult import RichResult
from ._sci_core import minimize

__all__ = ["fit", "profile_beta", "auto_order", "aic", "aicc",
           "neighbours", "starting_models"]

# set while auto_order is traversing the model space: one
# optimiser pass per candidate instead of restarting to
# convergence, since only the AIC ranking is needed there.
_SEARCHING = [False]

MAX_P = MAX_Q = 5
MAX_SEASONAL = 2
ROOT_TOL = 1.001


def _filter_column(w, ar, ma):
    """Innovations and their variances for one differenced column."""
    T, R, r = _sa._state_space(ar, ma)
    P = _sa._initial_covariance(T, R, r)
    a = [0.0] * r
    v_out, f_out = [], []
    for t in range(len(w)):
        f = P[0][0]
        if f <= 0.0:
            raise ValueError("sarimax: non-positive prediction variance")
        v = w[t] - a[0]
        PZ = [P[i][0] for i in range(r)]
        a = [a[i] + PZ[i] * v / f for i in range(r)]
        P = [[P[i][j] - PZ[i] * PZ[j] / f for j in range(r)]
             for i in range(r)]
        v_out.append(v)
        f_out.append(f)
        a = [sum(T[i][j] * a[j] for j in range(r)) for i in range(r)]
        TP = [[sum(T[i][k] * P[k][j] for k in range(r))
               for j in range(r)] for i in range(r)]
        P = [[sum(TP[i][k] * T[j][k] for k in range(r)) + R[i] * R[j]
              for j in range(r)] for i in range(r)]
    return v_out, f_out


def _residual_column(w, ar, ma):
    """Conditional residuals with unit variances -- the cheap filter."""
    r = _sa.css(w, ar, ma, full=True)
    return r["residuals"], [1.0] * len(w)


def profile_beta(wy, wX, ar=(), ma=(), filter="exact"):
    r"""Exact GLS for :math:`\beta` given the ARIMA parameters.

    Both ``wy`` and the columns of ``wX`` must already be differenced.
    """
    n = len(wy)
    if any(len(c) != n for c in wX):
        raise ValueError("sarimax: regressor columns must match the "
                         "differenced series length %d" % n)
    if filter not in ("exact", "conditional"):
        raise ValueError("sarimax: filter must be 'exact' or "
                         "'conditional', got %r" % filter)
    _col = _filter_column if filter == "exact" else _residual_column
    vy, f = _col(wy, ar, ma)
    if not wX:
        ssq = sum(vy[t] * vy[t] / f[t] for t in range(n))
        return {"beta": [], "ssq": ssq, "v": vy, "f": f,
                "information": [],
                "sum_log_f": sum(math.log(v) for v in f)}
    for j, c in enumerate(wX):
        if max(abs(v) for v in c) <= 1e-12:
            raise ValueError(
                "sarimax: regressor %d is annihilated by the "
                "differencing operator (a linear trend vanishes under "
                "nabla, a seasonal dummy under nabla_s), so beta is "
                "not identified" % j)
    vx = [_col(c, ar, ma)[0] for c in wX]
    k = len(vx)
    A = [[sum(vx[i][t] * vx[j][t] / f[t] for t in range(n))
          for j in range(k)] for i in range(k)]
    b = [sum(vx[i][t] * vy[t] / f[t] for t in range(n))
         for i in range(k)]
    beta = [float(v) for v in np.linalg.solve(np.array(A), np.array(b))]
    resid = [vy[t] - sum(beta[i] * vx[i][t] for i in range(k))
             for t in range(n)]
    ssq = sum(resid[t] * resid[t] / f[t] for t in range(n))
    return {"beta": beta, "ssq": ssq, "v": resid, "f": f,
            "information": A,
            "sum_log_f": sum(math.log(v) for v in f)}


def _columns(X, n):
    if X is None:
        return []
    cols = [list(c) for c in X] if not hasattr(X[0], "__len__") \
        else [[float(row[j]) for row in X] for j in range(len(X[0]))]
    if not hasattr(X[0], "__len__"):
        cols = [[float(v) for v in X]]
    for c in cols:
        if len(c) != n:
            raise ValueError("sarimax: regressor has %d rows but the "
                             "series has %d" % (len(c), n))
    return cols


def fit(y, X=None, order=(0, 1, 1), seasonal_order=(0, 1, 1), s=12,
        include_constant=None, method="ml"):
    r"""Fit a regression with seasonal ARIMA errors."""
    if method not in ("ml", "uls", "css"):
        raise ValueError("sarimax: method must be 'ml', 'uls' or "
                         "'css', got %r" % method)
    y = [float(v) for v in y]
    p, d, q = (int(v) for v in order)
    P, D, Q = (int(v) for v in seasonal_order)
    s = int(s)
    cols = _columns(X, len(y))
    if include_constant is None:
        include_constant = (d + D) < 2
    if include_constant and (d + D) >= 2:
        raise ValueError("sarimax: a constant is admitted only when "
                         "d + D < 2 (Hyndman-Khandakar 2008 Sec. 3.1), "
                         "got d = %d, D = %d" % (d, D))
    if include_constant:
        cols = [[1.0] * len(y)] + cols
    wy = _sa.difference(y, d, D, s)
    wX = [_sa.difference(c, d, D, s) for c in cols]
    if include_constant:
        wX[0] = [1.0] * len(wy)
    npar = p + q + P + Q
    if npar == 0 and not wX:
        raise ValueError("sarimax: nothing to estimate")

    def unpack(v):
        i = 0
        phi = list(v[i:i + p]); i += p
        th = list(v[i:i + q]); i += q
        Ph = list(v[i:i + P]); i += P
        return phi, th, Ph, list(v[i:i + Q])

    def objective(v):
        phi, th, Ph, Th = unpack(v)
        if not (_sa._roots_ok(phi, ROOT_TOL)
                and _sa._roots_ok(Ph, ROOT_TOL)):
            return 1e10
        ar, ma = _sa.expand_polynomials(phi, Ph, th, Th, s)
        if not _sa._roots_ok(ma, ROOT_TOL):
            return 1e10
        try:
            r = profile_beta(wy, wX, ar, ma,
                             "conditional" if method == "css"
                             else "exact")
        except (ValueError, ZeroDivisionError):
            return 1e10
        n = len(wy)
        s2 = r["ssq"] / n
        if s2 <= 0.0:
            return 1e10
        if method in ("uls", "css"):
            return r["ssq"]
        return (0.5 * n * (math.log(2.0 * math.pi * s2) + 1.0)
                + 0.5 * r["sum_log_f"])

    x0 = [0.1] * npar
    best, xhat = objective(x0), list(x0)
    res = None
    if npar:
        for _ in range(1 if _SEARCHING[0] else 8):
            res = minimize(objective, xhat, method="Nelder-Mead")
            cand = list(res.x if hasattr(res, "x") else res["x"])
            val = objective(cand)
            if val < best - 1e-11:
                best, xhat = val, cand
            else:
                break
    phi, th, Ph, Th = unpack(xhat)
    ar, ma = _sa.expand_polynomials(phi, Ph, th, Th, s)
    r = profile_beta(wy, wX, ar, ma)
    n = len(wy)
    sigma2 = r["ssq"] / n
    ll = (-0.5 * n * (math.log(2.0 * math.pi * sigma2) + 1.0)
          - 0.5 * r["sum_log_f"])
    k = npar + len(r["beta"]) + 1
    if r["beta"]:
        cov = np.linalg.inv(np.array(r["information"]))
        beta_se = [math.sqrt(max(sigma2 * float(cov[i][i]), 0.0))
                   for i in range(len(r["beta"]))]
    else:
        beta_se = []
    return RichResult(payload={
        "beta_se": beta_se,
        "estimate": r["beta"][0] if r["beta"] else sigma2,
        "beta": r["beta"], "phi": phi, "theta": th,
        "Phi": Ph, "Theta": Th, "ar": ar, "ma": ma,
        "sigma2": sigma2, "loglik": ll,
        "aic": -2.0 * ll + 2.0 * k, "n_par": k, "n_used": n,
        "residuals": r["v"], "innovation_variance": r["f"],
        "include_constant": bool(include_constant),
        "order": (p, d, q), "seasonal_order": (P, D, Q), "s": s,
        "fit_method": method,
        "method": "regression with seasonal ARIMA errors, beta "
                  "profiled out by exact GLS; Box et al. (2016) "
                  "Sec. 9.5, Hyndman & Khandakar (2008) Sec. 3.1",
    })


def aic(loglik, n_par):
    r"""Sec. 3.1: :math:`-2\log L + 2(p+q+P+Q+k)`."""
    return -2.0 * float(loglik) + 2.0 * int(n_par)


def aicc(loglik, n_par, n):
    r"""The small-sample corrected form used by the same package."""
    k, n = int(n_par), int(n)
    if n - k - 1 <= 0:
        return float("inf")
    return aic(loglik, k) + 2.0 * k * (k + 1) / float(n - k - 1)


def starting_models(d, D, s):
    r"""Step 1: the four models the search starts from."""
    if int(s) > 1:
        out = [((2, d, 2), (1, D, 1)), ((0, d, 0), (0, D, 0)),
               ((1, d, 0), (1, D, 0)), ((0, d, 1), (0, D, 1))]
    else:
        out = [((2, d, 2), (0, D, 0)), ((0, d, 0), (0, D, 0)),
               ((1, d, 0), (0, D, 0)), ((0, d, 1), (0, D, 0))]
    return out


def neighbours(order, seasonal_order, constant, s):
    r"""Step 2: the thirteen variations on the current model."""
    p, d, q = order
    P, D, Q = seasonal_order
    out = []
    for dp, dq, dP, dQ in ((1, 0, 0, 0), (-1, 0, 0, 0),
                           (0, 1, 0, 0), (0, -1, 0, 0),
                           (0, 0, 1, 0), (0, 0, -1, 0),
                           (0, 0, 0, 1), (0, 0, 0, -1),
                           (1, 1, 0, 0), (-1, -1, 0, 0),
                           (0, 0, 1, 1), (0, 0, -1, -1)):
        np_, nq = p + dp, q + dq
        nP, nQ = P + dP, Q + dQ
        if min(np_, nq, nP, nQ) < 0:
            continue
        if np_ > MAX_P or nq > MAX_Q:
            continue
        if nP > MAX_SEASONAL or nQ > MAX_SEASONAL:
            continue
        if int(s) <= 1 and (nP or nQ):
            continue
        out.append(((np_, d, nq), (nP, D, nQ), constant))
    out.append((tuple(order), tuple(seasonal_order), not constant))
    return out


def _try_fit(y, X, order, seasonal_order, s, constant, method):
    if constant and (order[1] + seasonal_order[1]) >= 2:
        return None
    if order[0] + order[2] + seasonal_order[0] + seasonal_order[2] == 0 \
            and not constant and X is None:
        return None
    try:
        return fit(y, X, order, seasonal_order, s,
                   include_constant=constant, method=method)
    except (ValueError, ZeroDivisionError, ArithmeticError):
        return None


def auto_order(y, X=None, d=0, D=0, s=1, method="css", max_steps=20):
    r"""The step-wise search of Hyndman & Khandakar (2008) Sec. 3.2.

    ``d`` and ``D`` are taken as given: the paper selects them with
    KPSS and Canova-Hansen tests, which are not implemented here.
    ``method="css"`` keeps the search affordable -- the conditional
    residual recursion rather than the Kalman filter -- and the winner
    is refitted before it is returned. Pass ``method="ml"`` to search
    on the exact likelihood instead, at roughly fifty times the cost.
    """
    d, D, s = int(d), int(D), int(s)
    visited = {}
    tried = []

    def score(order, seasonal, constant):
        key = (order, seasonal, bool(constant))
        if key in visited:
            return visited[key]
        r = _try_fit(y, X, order, seasonal, s, constant, method)
        visited[key] = r
        tried.append({"order": order, "seasonal_order": seasonal,
                      "constant": bool(constant),
                      "aic": None if r is None else r["aic"],
                      "rejected": r is None})
        return r

    constant = (d + D) < 2
    best = None
    _SEARCHING[0] = True
    for order, seasonal in starting_models(d, D, s):
        r = score(order, seasonal, constant)
        if r is not None and (best is None or r["aic"] < best[0]["aic"]):
            best = (r, order, seasonal, constant)
    if best is None:
        raise ValueError("sarimax: every starting model was rejected")
    steps = 0
    while steps < int(max_steps):
        steps += 1
        improved = False
        cur, order, seasonal, constant = best
        for o, so, c in neighbours(order, seasonal, constant, s):
            r = score(o, so, c)
            if r is not None and r["aic"] < cur["aic"] - 1e-8:
                best = (r, o, so, c)
                improved = True
                break
        if not improved:
            break
    _SEARCHING[0] = False
    r, order, seasonal, constant = best
    final = _try_fit(y, X, order, seasonal, s, constant, method)
    if final is not None:
        r = final
    return RichResult(payload={
        "estimate": r["aic"], "aic": r["aic"], "fit": r,
        "order": order, "seasonal_order": seasonal,
        "constant": bool(constant), "steps": steps,
        "n_models_tried": len(tried), "tried": tried,
        "s": s, "search_method": method,
        "differencing_note": "d and D are inputs; Hyndman & Khandakar "
                             "select them with KPSS and Canova-Hansen "
                             "tests, which are not implemented here",
        "method": "step-wise order selection; Hyndman & Khandakar "
                  "(2008) Sec. 3.2",
    })


def cheatsheet():
    return ("sarimax: y = beta'x + n with seasonal ARIMA errors. beta "
            "is profiled out by exact GLS on the Kalman innovations, "
            "so only the ARIMA parameters go to the optimiser. "
            "auto_order is Hyndman-Khandakar's step-wise search: four "
            "starting models, thirteen neighbours, AIC, and the four "
            "stated constraints (p,q<=5, P,Q<=2, root >= 1.001, drop "
            "anything that will not fit). d and D are inputs -- the "
            "KPSS and Canova-Hansen tests that choose them are not "
            "implemented.")


# compact alias per ledger/NAMING.md
sarimax = fit
