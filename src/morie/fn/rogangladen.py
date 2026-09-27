"""Prevalence from an imperfect test: MLE under known sensitivity and specificity.

Bilder & Loughin (2025), Analysis of Categorical Data with R, eq (6.2).
"""

import math

from ._richresult import RichResult
from ._rrng_core import qnorm

__all__ = ["rogangladen"]


def rogangladen(w, n, se, sp, alpha=0.05):
    """Maximise L(pi) = [Se pi + (1-Sp)(1-pi)]^w [1 - Se pi - (1-Sp)(1-pi)]^(n-w) (6.2).

    The apparent positive rate is w/n; the unrestricted MLE is
    (w/n + Sp - 1)/(Se + Sp - 1) (Rogan & Gladen 1978), truncated to [0, 1]. The
    Wald interval uses se = sqrt(p(1-p)/n)/(Se + Sp - 1), p = w/n, also truncated.

    Parameters
    ----------
    w, n : int
        Positive test results and sample size.
    se, sp : float
        Sensitivity and specificity, with Se + Sp > 1.
    alpha : float

    Returns
    -------
    RichResult
        Keys: estimate, apparent, ci, se.

    References
    ----------
    Rogan, W. J. & Gladen, B. (1978). American Journal of Epidemiology 107, 71-76.
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R
    (2nd ed.). CRC Press. Eq (6.2).

    Examples
    --------
    >>> round(rogangladen(30, 200, 0.9, 0.95)["estimate"], 12)
    0.117647058824
    """
    if n <= 0 or not 0 <= w <= n or se + sp <= 1:
        raise ValueError("need 0 <= w <= n, n > 0 and Se + Sp > 1")
    p = w / n
    j = se + sp - 1
    est = min(1.0, max(0.0, (p + sp - 1) / j))
    s = math.sqrt(p * (1 - p) / n) / j
    z = qnorm(1 - alpha / 2)
    raw = (p + sp - 1) / j
    ci = (min(1.0, max(0.0, raw - z * s)), min(1.0, max(0.0, raw + z * s)))
    return RichResult(
        title="Prevalence with test error",
        summary_lines=[("estimate", est), ("apparent", p)],
        payload={"estimate": est, "apparent": p, "ci": ci, "se": s},
    )


def cheatsheet():
    return "rogangladen: prevalence MLE (p + Sp - 1)/(Se + Sp - 1) under test error. Bilder & Loughin eq (6.2)."
