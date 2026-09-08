# morie.fn -- Hedderich shelf, rebuilt from the book (rootcoder007/morie)
"""Odds ratio over a fixed interval of an interval-scaled predictor.

Source READ FROM THE CORPUS PDF, page rendered with ``pdftoppm``:
Hedderich, Sachs and Reynarowych, *Applied Statistics: Methods Using R*,
section 8.4.4, printed page 834, equation (8.63)::

    log psi(a, b) = g(x = b) - g(x = a)
                  = beta0 + beta1 b - beta0 - beta1 a
                  = beta1 (b - a)

    psi(a, b) = exp(beta1 (b - a))                              (8.63)

The intercept cancels, so the odds ratio for moving the predictor from
``a`` to ``b`` depends on ``beta1`` and the span alone.

A Wald interval is supplied when ``se`` is given: the log odds ratio is
``beta1 (b - a)`` with standard error ``|b - a| se(beta1)``, so the
interval is ``exp(beta1 (b-a) +/- z_{1-alpha/2} |b-a| se)``.  This is
(8.56) transported through the same linear map.

Book worked example, same page: from the Challenger data the book
estimates ``beta1 = -0.2322``; a temperature rise of 10 degF gives
``psi(10) = exp(-2.322) = 0.098`` and a drop of 10 degF gives
``psi(-10) = exp(2.322) = 10.2``.
"""

from __future__ import annotations

import math

from . import _tail1core as C
from ._richresult import RichResult

__all__ = ["logor"]


def logor(beta1, a=0.0, b=1.0, se=None, level=0.95):
    """Odds ratio for a change of the predictor from ``a`` to ``b``.

    Parameters
    ----------
    beta1 : float
        Logistic-regression slope for the predictor, on the logit scale.
    a, b : float
        Endpoints of the interval.  The default ``a=0, b=1`` gives the
        per-unit odds ratio ``exp(beta1)``.
    se : float, optional
        Standard error of ``beta1``.  If given, a Wald interval is
        returned.
    level : float
        Confidence level, strictly inside (0, 1).

    Returns
    -------
    RichResult
        Keys: ``or``, ``logor``, ``span``, and when ``se`` is given
        ``se_logor``, ``ci_low``, ``ci_high``, ``level``.
    """
    beta1 = float(beta1)
    a = float(a)
    b = float(b)
    if not math.isfinite(beta1) or not math.isfinite(a) or not math.isfinite(b):
        raise ValueError("beta1, a and b must be finite")
    if not (0.0 < float(level) < 1.0):
        raise ValueError("level must be strictly between 0 and 1")
    span = b - a
    lor = beta1 * span
    payload = {"or": math.exp(lor), "logor": lor, "span": span}
    summary = [("Odds ratio", payload["or"]), ("log odds ratio", lor), ("span (b - a)", span)]
    if se is not None:
        se = float(se)
        if not math.isfinite(se) or se <= 0.0:
            raise ValueError("se must be a finite positive number")
        se_lor = abs(span) * se
        z = C.qnorm(0.5 + float(level) / 2.0)
        lo = lor - z * se_lor
        hi = lor + z * se_lor
        payload.update(
            {
                "se_logor": se_lor,
                "ci_low": math.exp(lo),
                "ci_high": math.exp(hi),
                "level": float(level),
            }
        )
        summary.append(("CI", (payload["ci_low"], payload["ci_high"])))
    return RichResult(
        title="Odds ratio over an interval (Hedderich eq. 8.63)",
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet() -> str:
    return "logor(beta1, a, b, se=None): odds ratio exp(beta1*(b-a)) -- Hedderich eq. (8.63)."
