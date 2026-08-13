# morie.fn -- function file (rootcoder007/morie)
r"""Prophet: piecewise trend, Fourier seasonality, holidays.

A decomposable model, eq. (1):

.. math:: y(t) = g(t) + s(t) + h(t) + \epsilon_t,

trend plus seasonality plus holidays. Every component is a regressor,
so the whole thing is a linear model and fits like one -- which is the
paper's actual argument. It is not that the model is better than ARIMA;
it is that its parameters mean something an analyst can reason about, so
a domain expert without time-series training can adjust it.

**The trend is piecewise linear, and continuity is not free.** With
changepoints at :math:`s_j` and rate adjustments :math:`\delta`,

.. math:: g(t) = \big(k + a(t)^\top\delta\big)\,t
          + \big(m + a(t)^\top\gamma\big), \qquad
          a_j(t) = 1\{t \ge s_j\},

and the offsets must satisfy :math:`\gamma_j = -s_j\delta_j` or the
segments do not join. Drop that and the fitted curve jumps at every
changepoint -- it still fits, because least squares absorbs the jumps
into the residual, and it forecasts nonsense past the last changepoint.
The anchor checks the left and right limits agree at every changepoint,
exactly.

**Seasonality is a Fourier series, and its order is the real knob.**

.. math:: s(t) = \sum_{n=1}^{N}\Big(a_n\cos\frac{2\pi n t}{P}
          + b_n\sin\frac{2\pi n t}{P}\Big)

is exactly periodic with period :math:`P` by construction. Raising
:math:`N` buys the ability to fit a faster-changing seasonal shape and
costs overfitting; the paper's defaults are 10 for yearly and 3 for
weekly.

**Holidays are the reason the model exists at scale.** They are not
seasonal -- Easter moves, Thanksgiving is the fourth Thursday -- so no
periodic basis can represent them. Each holiday gets its own indicator,
optionally widened by a window of days either side.

References
----------
Taylor, S. J. & Letham, B. (2018) "Forecasting at Scale", *The American
Statistician* 72(1), 37-45, doi:10.1080/00031305.2017.1380080;
preprint *PeerJ Preprints* 5:e3190v2,
doi:10.7287/peerj.preprints.3190v2. Eq. (1)-(4), Secs. 3.1-3.3.

Harvey, A. C. & Peters, S. (1990) "Estimation procedures for structural
time series models", *Journal of Forecasting* 9(2), 89-108,
doi:10.1002/for.3980090203. The decomposable structural model this
follows.

Hyndman, R. J. & Athanasopoulos, G. (2021) *Forecasting: Principles and
Practice*, 3rd edn, OTexts. Fourier terms for seasonality.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["trend_matrix", "fourier_terms", "holiday_matrix",
           "prophet_design", "prophet_fit", "prophet_predict",
           "piecewise_trend"]

_EPS = 1e-12


def _changepoints(t, n_changepoints, changepoint_range=0.8,
                  changepoints=None):
    if changepoints is not None:
        return [float(v) for v in changepoints]
    n = len(t)
    hi = t[0] + changepoint_range * (t[-1] - t[0])
    m = int(n_changepoints)
    if m < 1:
        return []
    step = (hi - t[0]) / (m + 1)
    return [t[0] + step * (j + 1) for j in range(m)]


def piecewise_trend(t, k_rate, m_off, deltas, cps):
    r"""Eq. (4) with :math:`\gamma_j = -s_j\delta_j`.

    The gamma is what joins the segments; without it the curve jumps at
    every changepoint and least squares simply absorbs the jumps.
    """
    out = []
    for tv in t:
        a = [1.0 if tv >= s else 0.0 for s in cps]
        rate = k_rate + sum(a[j] * deltas[j] for j in range(len(cps)))
        off = m_off + sum(a[j] * (-cps[j] * deltas[j])
                          for j in range(len(cps)))
        out.append(rate * tv + off)
    return out


def trend_matrix(t, cps):
    r"""Design columns for :math:`k`, :math:`m` and each
    :math:`\delta_j`.

    The delta column is :math:`a_j(t)(t - s_j)`, which already carries
    the :math:`-s_j\delta_j` offset -- so continuity holds by
    construction rather than being imposed afterwards.
    """
    rows = []
    for tv in t:
        row = [tv, 1.0]
        for s in cps:
            row.append((tv - s) if tv >= s else 0.0)
        rows.append(row)
    return rows


def fourier_terms(t, period, order):
    r"""Cosine and sine pairs, exactly periodic with ``period``."""
    if period <= 0.0:
        raise ValueError("prphet: period must be positive, got %r"
                         % (period,))
    if order < 1:
        raise ValueError("prphet: order must be at least 1, got %d"
                         % order)
    rows = []
    for tv in t:
        row = []
        for n in range(1, int(order) + 1):
            ang = 2.0 * math.pi * n * tv / float(period)
            row.append(math.cos(ang))
            row.append(math.sin(ang))
        rows.append(row)
    return rows


def holiday_matrix(t, holidays, lower=0, upper=0):
    r"""One indicator per holiday, optionally widened by a window.

    Holidays cannot be seasonal -- Easter moves and Thanksgiving is the
    fourth Thursday -- so no periodic basis represents them.
    """
    names = sorted(holidays)
    rows = []
    for tv in t:
        row = []
        for nm in names:
            hit = 0.0
            for d in holidays[nm]:
                if d - lower <= tv <= d + upper:
                    hit = 1.0
                    break
            row.append(hit)
        rows.append(row)
    return rows, names


def prophet_design(t, cps, seasonalities=None, holidays=None,
                   holiday_window=(0, 0)):
    """Stack trend, seasonality and holiday columns into one design."""
    tm = trend_matrix(t, cps)
    cols = ["k", "m"] + ["delta_%d" % j for j in range(len(cps))]
    blocks = [tm]
    seas = seasonalities or []
    for (name, period, order) in seas:
        blocks.append(fourier_terms(t, period, order))
        for n in range(1, int(order) + 1):
            cols += ["%s_cos%d" % (name, n), "%s_sin%d" % (name, n)]
    hn = []
    if holidays:
        hm, hn = holiday_matrix(t, holidays, holiday_window[0],
                                holiday_window[1])
        blocks.append(hm)
        cols += ["holiday_%s" % v for v in hn]
    X = [[v for b in blocks for v in b[i]] for i in range(len(t))]
    return X, cols, hn


def prophet_fit(t, y, n_changepoints=10, changepoint_range=0.8,
                changepoints=None, seasonalities=None, holidays=None,
                holiday_window=(0, 0), changepoint_prior=0.05,
                ridge=1e-8):
    r"""Fit by penalised least squares.

    The Laplace prior :math:`\delta_j \sim \mathrm{Laplace}(0,\tau)`
    is an **L1** penalty of :math:`1/\tau` on the rate adjustments
    alone -- never on :math:`k`, :math:`m`, the seasonal coefficients or
    the holidays, which carry no shrinkage in the model. A penalty on
    the intercept would shrink the level itself.

    It has to be L1 and not L2: the sparsity IS the selection
    mechanism. A ridge penalty shrinks every :math:`\delta_j` and zeroes
    none, so every candidate changepoint stays nominally active and the
    "automatic selection" selects nothing. Solved by cyclic coordinate
    descent with soft-thresholding on the penalised coordinates.
    """
    tv = k.vec(t)
    yv = k.vec(y)
    n = len(tv)
    if len(yv) != n:
        raise ValueError("prphet: %d times but %d observations"
                         % (n, len(yv)))
    if n < 8:
        raise ValueError("prphet: need at least 8 observations, got %d"
                         % n)
    tau = float(changepoint_prior)
    if tau <= 0.0:
        raise ValueError("prphet: changepoint_prior must be positive, "
                         "got %r" % (changepoint_prior,))
    cps = _changepoints(tv, n_changepoints, changepoint_range,
                        changepoints)
    X, cols, hn = prophet_design(tv, cps, seasonalities, holidays,
                                 holiday_window)
    p = len(cols)
    pen = [0.0] * p
    for j, c in enumerate(cols):
        if c.startswith("delta_"):
            pen[j] = 1.0 / tau
    # Cyclic coordinate descent: an ordinary least-squares update on
    # the unpenalised coordinates, soft-thresholded on the deltas.
    XtX = [[sum(X[i][a] * X[i][b] for i in range(n))
            for b in range(p)] for a in range(p)]
    Xty = [sum(X[i][a] * yv[i] for i in range(n)) for a in range(p)]
    beta = [0.0] * p
    for _ in range(400):
        shift = 0.0
        for a in range(p):
            gaa = XtX[a][a] + ridge
            if gaa <= 0.0:
                continue
            r = Xty[a] - sum(XtX[a][b] * beta[b] for b in range(p)
                             if b != a)
            if pen[a] > 0.0:
                # soft threshold: this is what actually sets deltas to
                # EXACTLY zero, which ridge never does
                nb = (0.0 if abs(r) <= pen[a]
                      else (r - math.copysign(pen[a], r)) / gaa)
            else:
                nb = r / gaa
            shift = max(shift, abs(nb - beta[a]))
            beta[a] = nb
        if shift < 1e-12:
            break
    fitted = [sum(X[i][a] * beta[a] for a in range(p))
              for i in range(n)]
    resid = [yv[i] - fitted[i] for i in range(n)]
    named = {cols[a]: beta[a] for a in range(p)}
    deltas = [named["delta_%d" % j] for j in range(len(cps))]
    return RichResult(payload={
        "estimate": fitted, "fitted": fitted, "residual": resid,
        "coef": named, "beta": beta, "columns": cols,
        "changepoints": cps, "deltas": deltas,
        "k": named["k"], "m": named["m"],
        "trend": piecewise_trend(tv, named["k"], named["m"], deltas,
                                 cps),
        "holiday_names": hn, "t": list(tv), "n": n,
        "changepoint_prior": tau,
        "n_active_changepoints": sum(1 for d in deltas if d != 0.0),
        "sigma": math.sqrt(sum(v * v for v in resid) / max(n - p, 1)),
        "seasonalities": [s[0] for s in (seasonalities or [])],
        "method": "Prophet decomposable model, Taylor & Letham (2018) "
                  "eq. (1) and (4)",
    })


def prophet_predict(fit, t_new, seasonalities=None, holidays=None,
                    holiday_window=(0, 0)):
    """Forecast at new times, reusing the fitted coefficients."""
    tn = k.vec(t_new)
    X, cols, _ = prophet_design(tn, fit["changepoints"], seasonalities,
                                holidays, holiday_window)
    if cols != fit["columns"]:
        raise ValueError("prphet: the prediction design does not match "
                         "the fitted one; pass the same seasonalities "
                         "and holidays")
    beta = fit["beta"]
    return [sum(X[i][a] * beta[a] for a in range(len(beta)))
            for i in range(len(tn))]


def cheatsheet():
    return ("prphet: y = g(t) + s(t) + h(t) + eps. Trend g = (k + "
            "a(t)'delta)t + (m + a(t)'gamma) with gamma_j = -s_j "
            "delta_j -- that is what JOINS the segments; without it the "
            "curve jumps at every changepoint and least squares hides "
            "it in the residual. s(t) is a Fourier series, exactly "
            "periodic. Holidays need their own indicators because they "
            "move. Penalise the deltas ONLY.")


# compact alias per ledger/NAMING.md
prophetfit = prophet_fit
