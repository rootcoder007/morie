# morie.fn -- Hedderich shelf, rebuilt from the book (rootcoder007/morie)
"""Deviance and Pearson goodness of fit for a Poisson regression.

Source READ FROM THE CORPUS PDF, page rendered with ``pdftoppm``:
Hedderich, Sachs and Reynarowych, *Applied Statistics: Methods Using R*,
section 8.5 "Poisson Regression and Loglinear Models", printed page 843,
equations (8.80) and (8.81)::

    D = 2 [ sum (y_i log y_i - y_i) - sum (y_i log lamhat_i - lamhat_i) ]
      = 2 sum [ y_i log(y_i / lamhat_i) - (y_i - lamhat_i) ]         (8.80)
      = 2 sum   y_i log(y_i / lamhat_i)

The three printed lines agree only when ``sum (y_i - lamhat_i) = 0``,
which the book states one paragraph above holds exactly when the model
carries an intercept.  The middle line is the general one and is what is
computed here; the third line is reported as ``D_nointercept`` so the
identity can be inspected, together with the residual ``sum(y - lamhat)``.

The book adds that D is asymptotically chi-squared on ``nu = n - p - 1``
degrees of freedom and can be approximated by the Pearson statistic
(8.81).

BOOK ERRATUM in (8.81): the printed statistic is
``D ~= sum (y_i - lamhat_i) / lamhat_i``.  That numerator is not squared,
so with an intercept the statistic sums to (nearly) zero by the identity
just quoted and cannot approximate a positive deviance.  The Pearson
chi-squared for a Poisson fit is ``sum (y_i - lamhat_i)^2 / lamhat_i``,
which is what is computed; the literal printed form is not implemented.

``0 log 0`` is taken as 0, its limiting value.
"""

from __future__ import annotations

import math

from . import _array_core as np
from . import _tail1core as C
from ._richresult import hypothesis_test_result

__all__ = ["poisdev"]


def poisdev(y, lamhat, p=None, alpha=0.05):
    """Poisson deviance, Pearson chi-squared and the goodness-of-fit test.

    Parameters
    ----------
    y : array-like
        Observed non-negative counts.
    lamhat : array-like
        Fitted Poisson means, strictly positive.
    p : int, optional
        Number of predictors excluding the intercept.  When given, the
        deviance is referred to chi-squared on ``n - p - 1`` degrees of
        freedom and a p-value and reject decision are returned.
    alpha : float
        Significance level for the decision.

    Returns
    -------
    RichResult
        statistic = D.  Keys: ``deviance``, ``pearson_chisq``,
        ``D_nointercept``, ``resid_sum``, ``df``, ``pvalue``, ``reject``.
    """
    y = [float(v) for v in np.asarray(y, dtype=float).ravel()]
    lam = [float(v) for v in np.asarray(lamhat, dtype=float).ravel()]
    n = len(y)
    if n == 0:
        raise ValueError("y must not be empty")
    if len(lam) != n:
        raise ValueError("y and lamhat must have the same length")
    for v in y:
        if v < 0.0 or not math.isfinite(v):
            raise ValueError("counts must be finite and non-negative")
    for v in lam:
        if not math.isfinite(v) or v <= 0.0:
            raise ValueError("every fitted mean must be finite and positive")
    d = 0.0
    d3 = 0.0
    chi = 0.0
    rs = 0.0
    for i in range(n):
        t = 0.0 if y[i] == 0.0 else y[i] * math.log(y[i] / lam[i])
        d += 2.0 * (t - (y[i] - lam[i]))
        d3 += 2.0 * t
        chi += (y[i] - lam[i]) ** 2 / lam[i]
        rs += y[i] - lam[i]
    df = None
    pval = float("nan")
    rej = None
    if p is not None:
        df = n - int(p) - 1
        if df < 1:
            raise ValueError("n - p - 1 must be at least 1")
        pval = 1.0 - C.pchisq(d, df)
        rej = bool(pval < float(alpha))
    return hypothesis_test_result(
        test_name="Poisson deviance goodness of fit (Hedderich eq. 8.80)",
        statistic=d,
        pvalue=pval,
        df=df,
        alpha=float(alpha),
        extra_summary=[("Pearson chi-squared", chi)],
        extra_payload={
            "deviance": d,
            "pearson_chisq": chi,
            "D_nointercept": d3,
            "resid_sum": rs,
            "reject": rej,
            "n": n,
        },
        callable_name="poisdev",
    )


def cheatsheet() -> str:
    return "poisdev(y, lamhat, p): Poisson deviance + Pearson chi-squared -- Hedderich eq. (8.80)."
