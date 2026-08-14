# morie.fn -- function file (rootcoder007/morie)
r"""Seasonal ARIMA: the multiplicative :math:`(p,d,q)\times(P,D,Q)_s` model.

**The model.** Two time intervals matter in periodic data -- adjacent
observations, and observations one period apart -- so the operators
multiply rather than add:

.. math:: \phi_p(B)\Phi_P(B^s)\nabla^d\nabla_s^D z_t
          = \theta_q(B)\Theta_Q(B^s) a_t.

The airline model :math:`(0,1,1)\times(0,1,1)_{12}` writes out as

.. math:: z_t - z_{t-1} - z_{t-12} + z_{t-13}
          = a_t - \theta a_{t-1} - \Theta a_{t-12} + \theta\Theta a_{t-13},

an MA(13) in :math:`w_t = \nabla\nabla_{12} z_t` with only **two** free
parameters. Its autocovariances vanish except at lags 1, 11, 12 and 13,
and

.. math:: \rho_1 = \frac{-\theta}{1+\theta^2}, \qquad
          \rho_{12} = \frac{-\Theta}{1+\Theta^2},

so :math:`\rho_1` is untouched by the seasonal factor and
:math:`\rho_{12}` by the regular one. That is what makes the two
parameters separately identifiable from the sample autocorrelations,
and it is the basis of ``preliminary_estimates``.

**Three estimation routes, all kept.** The book fits the airline data
three ways and they do not agree to the last decimal, so all three are
here and the difference is reported rather than hidden.

``moment``
    Solve :math:`\rho_1 = -\theta/(1+\theta^2)` and the seasonal
    counterpart. Cheap, used as a starting value. On the logged airline
    data :math:`r_1 = -0.34` and :math:`r_{12} = -0.39` give
    :math:`\hat\theta \simeq 0.39`, :math:`\hat\Theta \simeq 0.48`.

``css``
    Conditional sum of squares: run the residual recursion forward from
    zero starting values and minimise :math:`\sum a_t^2`. This is the
    unconditional least squares of Sec. 9.2.4 without the back-forecast
    correction; the book's iterative linearisation from these gives
    :math:`\hat\theta = 0.40 \pm 0.08`, :math:`\hat\Theta = 0.61 \pm
    0.07`.

``uls``
    The **exact** (unconditional) sum of squares
    :math:`S(\theta,\Theta) = \sum [a_t]^2` of Sec. 9.2.4, obtained
    from the same Kalman recursion as ``ml`` but without the
    :math:`\log|F|` term -- these are the contours of Figure 9.7. The
    book's headline least-squares estimates come from this surface:
    :math:`\hat\theta = 0.40 \pm 0.08`, :math:`\hat\Theta = 0.61
    \pm 0.07`, :math:`\hat\sigma_a^2 = 1.34\times10^{-3}`.

``ml`` (default)
    The exact Gaussian likelihood through a Kalman filter on the
    state-space form, which is what ``stats::arima`` uses and what the
    book reports in the R output of Sec. 9.2.4: ``ma1 = -0.4018``,
    ``sma1 = -0.5569``, ``sigma^2 = 0.001348``, ``log likelihood =
    244.7``, ``aic = -483.4``. The anchor reproduces all five numbers.

The exact likelihood is the default because it is the only one of the
three that uses the first :math:`q` observations correctly; the initial
state covariance is the stationary solution of
:math:`P = TPT' + RR'`, not :math:`RR'`. Getting that wrong costs
about 0.03 in :math:`\hat\theta` and moves the likelihood in the wrong
direction, which is exactly the kind of error a parity check between
two of my own routes would not catch.

**Sign convention.** The book writes the moving average operator with
minus signs, ``(1 - theta B)``; R writes it with plus signs. The
estimates therefore print with opposite signs in the two places. This
module follows the book, and ``r_convention`` returns the negated
coefficients for comparison with ``stats::arima`` output.

References
----------
Box, G. E. P., Jenkins, G. M., Reinsel, G. C. & Ljung, G. M. (2016)
*Time Series Analysis: Forecasting and Control*, 5th edn, Wiley,
ISBN 978-1-118-67502-1. Chapter 9 throughout: Sec. 9.1.3 for the
general multiplicative model (9.1.7) and the order notation
:math:`(p,d,q)\times(P,D,Q)_s`; Sec. 9.2.1 for the airline model
(9.2.1)-(9.2.2) and its invertibility region; Sec. 9.2.2 for the
difference-equation forecasts (9.2.3)-(9.2.6); Sec. 9.2.3 for the
autocovariances (9.2.18), the closed forms for :math:`\rho_1` and
:math:`\rho_{12}`, Bartlett's variance (9.2.19) and the preliminary
estimates :math:`\hat\theta \simeq 0.39`, :math:`\hat\Theta \simeq
0.48` from :math:`r_1 = -0.34`, :math:`r_{12} = -0.39`; Sec. 9.2.4 for
the conditional recursion (9.2.20), the least-squares estimates
:math:`0.40 \pm 0.08` and :math:`0.61 \pm 0.07` with
:math:`\hat\sigma_a^2 = 1.34\times10^{-3}`, the large-sample variances
(9.2.21), and the R output quoted above; and Part Five, Series G, for
the 144 monthly airline passenger totals reproduced in
``series_g``.

Harvey, A. C. (1989) *Forecasting, Structural Time Series Models and
the Kalman Filter*, Cambridge University Press,
doi:10.1017/CBO9781107049994, Sec. 3.3, for the state-space form of an
ARMA process used by ``loglik`` and for the stationary initial state
covariance.
"""

import math

from . import _array_core as np
from ._richresult import RichResult
from ._sci_core import minimize

__all__ = ["series_g", "difference", "expand_polynomials",
           "sample_acf", "airline_autocovariances",
           "preliminary_estimates", "moment_estimate", "css",
           "loglik", "fit",
           "forecast", "large_sample_se", "bartlett_se",
           "r_convention"]

METHODS = ("ml", "uls", "css", "moment")

# Series G, monthly totals of international airline passengers in
# thousands, January 1949 - December 1960; Box et al. (2016) Part Five.
# Stored month-major exactly as the book prints the table.
_SERIES_G_BY_MONTH = (
    (112, 115, 145, 171, 196, 204, 242, 284, 315, 340, 360, 417),
    (118, 126, 150, 180, 196, 188, 233, 277, 301, 318, 342, 391),
    (132, 141, 178, 193, 236, 235, 267, 317, 356, 362, 406, 419),
    (129, 135, 163, 181, 235, 227, 269, 313, 348, 348, 396, 461),
    (121, 125, 172, 183, 229, 234, 270, 318, 355, 363, 420, 472),
    (135, 149, 178, 218, 243, 264, 315, 374, 422, 435, 472, 535),
    (148, 170, 199, 230, 264, 302, 364, 413, 465, 491, 548, 622),
    (148, 170, 199, 242, 272, 293, 347, 405, 467, 505, 559, 606),
    (136, 158, 184, 209, 237, 259, 312, 355, 404, 404, 463, 508),
    (119, 133, 162, 191, 211, 229, 274, 306, 347, 359, 407, 461),
    (104, 114, 146, 172, 180, 203, 237, 271, 305, 310, 362, 390),
    (118, 140, 166, 194, 201, 229, 278, 306, 336, 337, 405, 432),
)


def series_g(log=False):
    r"""The 144 airline passenger totals, in calendar order."""
    out = [float(_SERIES_G_BY_MONTH[m][y])
           for y in range(12) for m in range(12)]
    return [math.log(v) for v in out] if log else out


def difference(y, d=0, D=0, s=1):
    r"""Apply :math:`\nabla^d\nabla_s^D`, regular first then seasonal."""
    d, D, s = int(d), int(D), int(s)
    if d < 0 or D < 0:
        raise ValueError("sarima: d and D must be non-negative")
    if D and s < 2:
        raise ValueError("sarima: seasonal differencing needs s >= 2, "
                         "got %d" % s)
    w = [float(v) for v in y]
    for _ in range(d):
        if len(w) < 2:
            raise ValueError("sarima: series too short to difference")
        w = [w[t] - w[t - 1] for t in range(1, len(w))]
    for _ in range(D):
        if len(w) <= s:
            raise ValueError("sarima: series too short for seasonal "
                             "differencing at s = %d" % s)
        w = [w[t] - w[t - s] for t in range(s, len(w))]
    return w


def _poly_mult(a, b):
    """Multiply two polynomials given as coefficient lists in B."""
    out = [0.0] * (len(a) + len(b) - 1)
    for i, u in enumerate(a):
        for j, v in enumerate(b):
            out[i + j] += u * v
    return out


def _seasonal_lift(c, s):
    """Turn a polynomial in :math:`B^s` into one in :math:`B`."""
    out = [0.0] * ((len(c) - 1) * s + 1)
    for i, v in enumerate(c):
        out[i * s] = v
    return out


def expand_polynomials(phi=(), Phi=(), theta=(), Theta=(), s=12):
    r"""Multiply out the seasonal and regular operators.

    Returns the AR and MA coefficient lists of the equivalent
    non-seasonal ARMA on the differenced series, in the book's sign
    convention: :math:`w_t = \sum ar_i w_{t-i} + a_t
    - \sum ma_j a_{t-j}`.
    """
    s = int(s)
    if (Phi or Theta) and s < 2:
        raise ValueError("sarima: seasonal terms need s >= 2, got %d" % s)
    ar_poly = _poly_mult([1.0] + [-float(v) for v in phi],
                         _seasonal_lift([1.0] + [-float(v) for v in Phi],
                                        s))
    ma_poly = _poly_mult([1.0] + [-float(v) for v in theta],
                         _seasonal_lift([1.0] + [-float(v) for v in Theta],
                                        s))
    return ([-v for v in ar_poly[1:]], [-v for v in ma_poly[1:]])


def sample_acf(x, lags):
    r"""Sample autocorrelations at the requested lags."""
    n = len(x)
    if n < 2:
        raise ValueError("sarima: need at least two observations")
    m = sum(x) / float(n)
    d = sum((v - m) ** 2 for v in x)
    if d <= 0.0:
        raise ValueError("sarima: the series is constant")
    out = {}
    for k in lags:
        k = int(k)
        if k < 1 or k >= n:
            raise ValueError("sarima: lag %d out of range" % k)
        out[k] = sum((x[t] - m) * (x[t - k] - m)
                     for t in range(k, n)) / d
    return out


def airline_autocovariances(theta, Theta, sigma2=1.0):
    r"""Equation (9.2.18): the only nonzero lags are 1, 11, 12, 13."""
    th, TH = float(theta), float(Theta)
    g = {0: (1.0 + th * th) * (1.0 + TH * TH) * sigma2,
         1: -th * (1.0 + TH * TH) * sigma2,
         11: th * TH * sigma2,
         12: -TH * (1.0 + th * th) * sigma2,
         13: th * TH * sigma2}
    return {"gamma": g,
            "rho": {k: v / g[0] for k, v in g.items()},
            "rho_1": -th / (1.0 + th * th),
            "rho_12": -TH / (1.0 + TH * TH),
            "nonzero_lags": (1, 11, 12, 13)}


def _invert_rho(rho):
    r"""Solve :math:`\rho = -x/(1+x^2)` for the invertible root."""
    r = float(rho)
    if abs(r) > 0.5:
        raise ValueError("sarima: |rho| = %.4f exceeds 0.5, so no "
                         "invertible MA(1) reproduces it" % abs(r))
    disc = math.sqrt(1.0 - 4.0 * r * r)
    return (-1.0 + disc) / (2.0 * r) if r != 0.0 else 0.0


def moment_estimate(rho):
    r"""The invertible MA(1) parameter matching a given
    :math:`\rho`.

    Solves :math:`\rho = -x/(1+x^2)`, which is where the book's
    :math:`\hat\theta \simeq 0.39` and :math:`\hat\Theta \simeq
    0.48` come from -- fed the *printed* :math:`r_1 = -0.34` and
    :math:`r_{12} = -0.39`. From the unrounded sample values the same
    solve returns 0.394 and 0.473; the difference is the rounding in
    the book, not a different estimator.
    """
    return _invert_rho(rho)


def preliminary_estimates(w, s=12):
    r"""Moment estimates from :math:`r_1` and :math:`r_s`."""
    r = sample_acf(w, (1, int(s)))
    th = _invert_rho(r[1])
    TH = _invert_rho(r[int(s)])
    return RichResult(payload={
        "estimate": th, "theta": th, "Theta": TH,
        "r_1": r[1], "r_s": r[int(s)],
        "method": "moments from rho_1 and rho_s; Box et al. (2016) "
                  "Sec. 9.2.3",
    })


def css(w, ar=(), ma=(), full=False):
    r"""Conditional sum of squares, equation (9.2.20).

    The residual recursion is run forward from zero starting values,
    so the first observations are conditioned on rather than modelled.
    """
    ar = [float(v) for v in ar]
    ma = [float(v) for v in ma]
    n = len(w)
    if n == 0:
        raise ValueError("sarima: no observations")
    a = [0.0] * n
    ssq = 0.0
    for t in range(n):
        pred = 0.0
        for i, c in enumerate(ar):
            if t - i - 1 >= 0:
                pred += c * w[t - i - 1]
        for j, c in enumerate(ma):
            if t - j - 1 >= 0:
                pred -= c * a[t - j - 1]
        a[t] = w[t] - pred
        ssq += a[t] * a[t]
    if full:
        return {"ssq": ssq, "residuals": a, "sigma2": ssq / float(n)}
    return ssq


def _state_space(ar, ma):
    """Harvey's ARMA state space; returns (T, R, r)."""
    p, q = len(ar), len(ma)
    r = max(p, q + 1)
    T = [[0.0] * r for _ in range(r)]
    for i in range(r - 1):
        T[i][i + 1] = 1.0
    for i in range(p):
        T[i][0] = ar[i]
    R = [1.0] + [-v for v in ma] + [0.0] * (r - q - 1)
    return T, R[:r], r


def _initial_covariance(T, R, r):
    r"""Stationary solution of :math:`P = TPT' + RR'`.

    Solved directly from :math:`(I - T\otimes T)\,\mathrm{vec}(P)
    = \mathrm{vec}(RR')`, which is exact for a stationary T and, for the
    pure moving average case, reduces to the finite sum
    :math:`\sum_i T^i RR' T'^i`.
    """
    n = r * r
    A = [[0.0] * n for _ in range(n)]
    b = [0.0] * n
    for i in range(r):
        for j in range(r):
            row = i * r + j
            A[row][row] += 1.0
            b[row] = R[i] * R[j]
            for k in range(r):
                for m in range(r):
                    A[row][k * r + m] -= T[i][k] * T[j][m]
    vec = np.linalg.solve(np.array(A), np.array(b))
    return [[float(vec[i * r + j]) for j in range(r)] for i in range(r)]


def loglik(w, ar=(), ma=()):
    r"""Exact Gaussian log-likelihood, concentrated on :math:`\sigma^2`.

    A Kalman filter over Harvey's state-space form. The initial state
    covariance is the stationary one; using :math:`RR'` instead quietly
    mis-weights the first :math:`q` observations.
    """
    ar = [float(v) for v in ar]
    ma = [float(v) for v in ma]
    n = len(w)
    if n == 0:
        raise ValueError("sarima: no observations")
    T, R, r = _state_space(ar, ma)
    P = _initial_covariance(T, R, r)
    a = [0.0] * r
    ssq = 0.0
    sumlogf = 0.0
    for t in range(n):
        f = P[0][0]
        if f <= 0.0:
            raise ValueError("sarima: non-positive prediction variance; "
                             "the parameters are outside the stationary "
                             "region")
        v = w[t] - a[0]
        PZ = [P[i][0] for i in range(r)]
        a = [a[i] + PZ[i] * v / f for i in range(r)]
        P = [[P[i][j] - PZ[i] * PZ[j] / f for j in range(r)]
             for i in range(r)]
        ssq += v * v / f
        sumlogf += math.log(f)
        a = [sum(T[i][j] * a[j] for j in range(r)) for i in range(r)]
        TP = [[sum(T[i][k] * P[k][j] for k in range(r)) for j in range(r)]
              for i in range(r)]
        P = [[sum(TP[i][k] * T[j][k] for k in range(r)) + R[i] * R[j]
              for j in range(r)] for i in range(r)]
    sigma2 = ssq / float(n)
    ll = (-0.5 * n * (math.log(2.0 * math.pi * sigma2) + 1.0)
          - 0.5 * sumlogf)
    return {"loglik": ll, "sigma2": sigma2, "n": n,
            "exact_ssq": ssq, "sum_log_f": sumlogf}


def _roots_ok(coefs, tol=1.001):
    """Reject a polynomial with a root inside ``tol`` (HK 2008 rule)."""
    if not coefs:
        return True
    poly = [1.0] + [-float(v) for v in coefs]
    while len(poly) > 1 and poly[-1] == 0.0:
        poly.pop()
    if len(poly) == 1:
        return True
    k = len(poly) - 1
    C = [[0.0] * k for _ in range(k)]
    for j in range(k):
        C[0][j] = -poly[j + 1] / poly[0]
    for i in range(1, k):
        C[i][i - 1] = 1.0
    ev = np.linalg.eigvals(np.array(C))
    for lam in ev:
        m = abs(complex(lam))
        if m <= 0.0:
            continue
        if 1.0 / m < tol:
            return False
    return True


def fit(y, order=(0, 1, 1), seasonal_order=(0, 1, 1), s=12,
        method="ml", start=None):
    r"""Fit a multiplicative seasonal ARIMA model.

    ``method`` is one of ``"ml"`` (exact likelihood, the default),
    ``"css"`` (conditional least squares) or ``"moment"`` (the
    :math:`r_1`, :math:`r_s` estimates, defined only for the airline
    model).
    """
    if method not in METHODS:
        raise ValueError("sarima: method must be one of %s, got %r"
                         % (", ".join(METHODS), method))
    p, d, q = (int(v) for v in order)
    P, D, Q = (int(v) for v in seasonal_order)
    s = int(s)
    if min(p, d, q, P, D, Q) < 0:
        raise ValueError("sarima: orders must be non-negative")
    w = difference(y, d, D, s)
    npar = p + q + P + Q
    if npar == 0:
        raise ValueError("sarima: the model has no free parameters")
    if len(w) <= npar:
        raise ValueError("sarima: %d differenced observations cannot "
                         "support %d parameters" % (len(w), npar))
    if method == "moment":
        if (p, q, P, Q) != (0, 1, 0, 1):
            raise ValueError("sarima: the moment route is defined for "
                             "the (0,d,1)x(0,D,1) airline model only, "
                             "got orders (%d,%d)x(%d,%d)"
                             % (p, q, P, Q))
        pre = preliminary_estimates(w, s)
        theta = [pre["theta"]]
        Theta = [pre["Theta"]]
        ar, ma = expand_polynomials((), (), theta, Theta, s)
        ll = loglik(w, ar, ma)
        return _package(y, w, (), theta, (), Theta, s, order,
                        seasonal_order, ll, css(w, ar, ma, full=True),
                        method, None)

    def unpack(v):
        i = 0
        phi = list(v[i:i + p]); i += p
        th = list(v[i:i + q]); i += q
        Ph = list(v[i:i + P]); i += P
        Th = list(v[i:i + Q])
        return phi, th, Ph, Th

    def objective(v):
        phi, th, Ph, Th = unpack(v)
        if not (_roots_ok(phi) and _roots_ok(Ph)):
            return 1e10
        ar, ma = expand_polynomials(phi, Ph, th, Th, s)
        if not _roots_ok(ma):
            return 1e10
        try:
            if method == "css":
                return css(w, ar, ma)
            if method == "uls":
                return loglik(w, ar, ma)["exact_ssq"]
            return -loglik(w, ar, ma)["loglik"]
        except ValueError:
            return 1e10

    if start is not None:
        x0 = [float(v) for v in start]
        if len(x0) != npar:
            raise ValueError("sarima: %d starting values for %d "
                             "parameters" % (len(x0), npar))
    elif (p, q, P, Q) == (0, 1, 0, 1):
        pre = preliminary_estimates(w, s)
        x0 = [pre["theta"], pre["Theta"]]
    else:
        x0 = [0.1] * npar
    # Nelder-Mead collapses its simplex around the starting point, so a
    # single call stops short of the optimum here: from the moment
    # estimates it returns loglik 244.09 where the optimum is 244.70.
    # Restart from the previous solution until it stops improving.
    res = None
    best = objective(x0)
    xhat = list(x0)
    for _ in range(8):
        res = minimize(objective, xhat, method="Nelder-Mead")
        cand = list(res.x if hasattr(res, "x") else res["x"])
        val = objective(cand)
        if val < best - 1e-11:
            best, xhat = val, cand
        else:
            xhat = cand if val < best else xhat
            break
    phi, th, Ph, Th = unpack(xhat)
    ar, ma = expand_polynomials(phi, Ph, th, Th, s)
    ll = loglik(w, ar, ma)
    return _package(y, w, phi, th, Ph, Th, s, order, seasonal_order,
                    ll, css(w, ar, ma, full=True), method, res)


def _package(y, w, phi, theta, Phi, Theta, s, order, seasonal_order,
             ll, cs, method, res):
    npar = len(phi) + len(theta) + len(Phi) + len(Theta)
    sigma2 = ll["sigma2"] if method in ("ml", "uls") else cs["sigma2"]
    aic = -2.0 * ll["loglik"] + 2.0 * (npar + 1)
    ar, ma = expand_polynomials(phi, Phi, theta, Theta, s)
    return RichResult(payload={
        "estimate": sigma2, "sigma2": sigma2,
        "phi": list(phi), "theta": list(theta),
        "Phi": list(Phi), "Theta": list(Theta),
        "ar": ar, "ma": ma,
        "loglik": ll["loglik"], "aic": aic,
        "n_used": len(w), "n_par": npar,
        "residuals": cs["residuals"], "ssq": cs["ssq"],
        "order": tuple(int(v) for v in order),
        "seasonal_order": tuple(int(v) for v in seasonal_order),
        "s": int(s), "y": [float(v) for v in y], "w": w,
        "fit_method": method,
        "converged": bool(getattr(res, "success", True))
        if res is not None else True,
        "method": "multiplicative seasonal ARIMA by %s; Box et al. "
                  "(2016) Ch. 9" % method,
    })


def forecast(fitted, h=12):
    r"""Difference-equation forecasts, equations (9.2.3)-(9.2.6).

    Unknown :math:`z`'s are replaced by their forecasts and unknown
    :math:`a`'s by zero; the known :math:`a`'s are the one-step-ahead
    errors already computed.
    """
    h = int(h)
    if h < 1:
        raise ValueError("sarima: h must be at least 1")
    y = list(fitted["y"])
    d, D = fitted["order"][1], fitted["seasonal_order"][1]
    s = fitted["s"]
    ar, ma = fitted["ar"], fitted["ma"]
    # the full operator on z, including the differencing factors
    lhs = _poly_mult([1.0] + [-float(v) for v in ar],
                     _poly_mult([1.0, -1.0] * 0 + _diff_poly(d, 1),
                                _diff_poly(D, s)))
    z_ar = [-v for v in lhs[1:]]
    a = list(fitted["residuals"])
    zpad = list(y)
    apad = [0.0] * (len(y) - len(a)) + list(a)
    out = []
    for step in range(h):
        t = len(zpad)
        val = 0.0
        for i, c in enumerate(z_ar):
            val += c * zpad[t - i - 1]
        for j, c in enumerate(ma):
            idx = t - j - 1
            if idx < len(apad):
                val -= c * apad[idx]
        zpad.append(val)
        apad.append(0.0)
        out.append(val)
    psi = _psi_weights(z_ar, ma, h)
    var = [fitted["sigma2"] * sum(p * p for p in psi[:i + 1])
           for i in range(h)]
    return RichResult(payload={
        "estimate": out[0], "forecast": out, "variance": var,
        "se": [math.sqrt(v) for v in var], "psi": psi,
        "method": "difference-equation forecasts; Box et al. (2016) "
                  "Sec. 9.2.2",
    })


def _diff_poly(k, s):
    """:math:`(1 - B^s)^k` as a coefficient list."""
    out = [1.0]
    for _ in range(int(k)):
        out = _poly_mult(out, [1.0] + [0.0] * (s - 1) + [-1.0])
    return out


def _psi_weights(ar, ma, h):
    r"""The :math:`\psi` weights of the forecast error, eq. (9.2.8)."""
    psi = [1.0]
    for j in range(1, h):
        v = -ma[j - 1] if j - 1 < len(ma) else 0.0
        for i, c in enumerate(ar):
            if j - i - 1 >= 0:
                v += c * psi[j - i - 1]
        psi.append(v)
    return psi


def large_sample_se(theta, Theta, n):
    r"""Equation (9.2.21) for the airline model.

    The off-diagonal term of the information matrix carries
    :math:`\theta^{11}`, so unless :math:`|\theta|` is near one the two
    estimates are effectively uncorrelated.
    """
    th, TH, n = float(theta), float(Theta), int(n)
    if n < 1:
        raise ValueError("sarima: n must be positive")
    v_th = (1.0 - th * th) / n
    v_TH = (1.0 - TH * TH) / n
    return {"var_theta": v_th, "var_Theta": v_TH,
            "se_theta": math.sqrt(max(v_th, 0.0)),
            "se_Theta": math.sqrt(max(v_TH, 0.0)),
            "cov": 0.0,
            "off_diagonal_term": th ** 11 / (1.0 - th ** 12 * TH)}


def bartlett_se(rho, n):
    r"""Equation (9.2.19): the standard error of high-lag :math:`r_k`.

    ``rho`` supplies the autocorrelations at lags 1, 11, 12 and 13; the
    remaining lags are zero under the airline model, which is why the
    formula is this short.
    """
    n = int(n)
    if n < 1:
        raise ValueError("sarima: n must be positive")
    ssq = sum(float(rho[k]) ** 2 for k in (1, 11, 12, 13))
    var = (1.0 + 2.0 * ssq) / n
    return {"variance": var, "se": math.sqrt(var),
            "white_noise_se": math.sqrt(1.0 / n)}


def r_convention(fitted):
    r"""The same fit printed the way ``stats::arima`` prints it.

    R writes the moving average operator with plus signs, so every MA
    coefficient appears negated relative to the book.
    """
    return {"ma": [-v for v in fitted["theta"]],
            "sma": [-v for v in fitted["Theta"]],
            "ar": list(fitted["phi"]), "sar": list(fitted["Phi"]),
            "sigma2": fitted["sigma2"], "loglik": fitted["loglik"],
            "aic": fitted["aic"],
            "note": "R writes (1 + theta B); the book writes "
                    "(1 - theta B)"}


def cheatsheet():
    return ("sarima: phi(B)Phi(B^s) nabla^d nabla_s^D z = "
            "theta(B)Theta(B^s) a. The airline (0,1,1)x(0,1,1)_12 is "
            "an MA(13) in w = nabla nabla_12 z with two parameters, "
            "nonzero autocorrelations only at lags 1, 11, 12, 13, and "
            "rho_1 = -theta/(1+theta^2) untouched by the seasonal "
            "factor. Three routes kept: moment, css, and the exact "
            "likelihood (default) -- on the logged airline data the "
            "last reproduces R's 0.4018 / 0.5569, sigma^2 0.001348, "
            "loglik 244.7, aic -483.4.")


# compact alias per ledger/NAMING.md
seasonal_arima = fit
