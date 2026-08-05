# morie.fn -- Hedderich shelf, rebuilt from the book (rootcoder007/morie)
"""Wald statistic and confidence interval for logistic-regression coefficients.

Source READ FROM THE CORPUS PDF, page rendered with ``pdftoppm``:
Hedderich, Sachs and Reynarowych, *Applied Statistics: Methods Using R*,
section 8.4, printed pages 829-830, equations (8.56) and (8.57)::

    beta_hat_i +/- z_{1-alpha/2} se(beta_hat_i)      for i = 0, 1   (8.56)

    W_hat = beta_hat_i / se(beta_hat_i)              for i = 0, 1   (8.57)

W_hat is asymptotically standard normal under H0: beta_i = 0, so the
two-sided p-value is ``2 (1 - Phi(|W|))``.

BOOK ERRATUM, confirmed by re-running the fit: the R input block printed
on page 829 lists ``t`` and ``d`` vectors that do not pair up -- fitting
the printed vectors gives beta0 = 23.775, beta1 = -0.3667, not the
values in the printed output.  The printed *output* on the same page is
nevertheless the correct Challenger O-ring logistic fit
(beta0 = 15.0429, se 7.3786, z 2.039; beta1 = -0.2322, se 0.1082,
z -2.145, p 0.0415 / 0.0320), reproduced here from the standard 23-flight
temperature/distress pairing.  The printed output is used as the anchor;
the printed input vectors are not.
"""

from __future__ import annotations

import math

from . import _array_core as np
from . import _tail1core as C
from ._richresult import RichResult

__all__ = ["lrwald"]


def lrwald(beta, se, level=0.95, alpha=0.05):
    """Wald z, p-value and confidence interval per coefficient.

    Parameters
    ----------
    beta : array-like
        Estimated coefficients.
    se : array-like
        Their standard errors; must be positive and the same length.
    level : float
        Confidence level for the interval (8.56).
    alpha : float
        Significance level used for the per-coefficient reject decision.

    Returns
    -------
    RichResult
        Keys: ``beta``, ``se``, ``z``, ``pvalue``, ``ci_low``,
        ``ci_high``, ``reject``, ``level``, ``alpha``.
    """
    beta = [float(v) for v in np.asarray(beta, dtype=float).ravel()]
    se = [float(v) for v in np.asarray(se, dtype=float).ravel()]
    if len(beta) == 0:
        raise ValueError("beta must not be empty")
    if len(beta) != len(se):
        raise ValueError("beta and se must have the same length")
    for v in se:
        if not math.isfinite(v) or v <= 0.0:
            raise ValueError("every standard error must be finite and positive")
    if not (0.0 < float(level) < 1.0):
        raise ValueError("level must be strictly between 0 and 1")
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError("alpha must be strictly between 0 and 1")
    z = C.qnorm(0.5 + float(level) / 2.0)
    stat = [b / s for b, s in zip(beta, se)]
    pval = [2.0 * (1.0 - C.pnorm(abs(w))) for w in stat]
    lo = [b - z * s for b, s in zip(beta, se)]
    hi = [b + z * s for b, s in zip(beta, se)]
    rej = [bool(p < float(alpha)) for p in pval]
    return RichResult(
        title="Wald test for logistic-regression coefficients (Hedderich eqs. 8.56, 8.57)",
        summary_lines=[("z", stat), ("p-value", pval)],
        payload={
            "beta": beta,
            "se": se,
            "z": stat,
            "pvalue": pval,
            "ci_low": lo,
            "ci_high": hi,
            "reject": rej,
            "level": float(level),
            "alpha": float(alpha),
        },
    )


def cheatsheet() -> str:
    return "lrwald(beta, se): Wald z, p and CI per coefficient -- Hedderich eqs. (8.56), (8.57)."
