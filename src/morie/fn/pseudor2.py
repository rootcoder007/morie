# morie.fn -- Hedderich shelf, rebuilt from the book (rootcoder007/morie)
"""Pseudo coefficients of determination for a logistic regression.

Source READ FROM THE CORPUS PDF, pages rendered with ``pdftoppm``:
Hedderich, Sachs and Reynarowych, *Applied Statistics: Methods Using R*,
section 8.4.6 "Pseudo Coefficients of Determination", printed pages
838-839, equations (8.70), (8.71) and (8.72)::

    R2_McF = (LL0 - LLmod) / LL0 = 1 - LLmod / LL0               (8.70)

    R2_CS  = 1 - (L0 / Lmod)^(2/n)                               (8.71)

    R2_N   = R2_CS / R2_CSmax = R2_CS / (1 - L0^(2/n))           (8.72)

LL is the log-likelihood and L = exp(LL) the likelihood; ``mod`` is the
fitted model, ``0`` the intercept-only null model, ``n`` the number of
observations the model was fitted on.  Written on the log scale here so
that L0 and Lmod never underflow:

    R2_CS = 1 - exp(2 (LL0 - LLmod) / n)
    R2_N  = R2_CS / (1 - exp(2 LL0 / n))

The book records the ranges: 0 <= R2_McF < 1 (below 0.2 low, above 0.4 a
significant improvement) and 0 <= R2_CS < 1, with R2_N normalised so
that 1 is attainable (above 0.2 acceptable, above 0.5 very good).
"""

from __future__ import annotations

import math

from ._richresult import RichResult

__all__ = ["pseudor2"]


def pseudor2(llmod, llnull, n):
    """McFadden, Cox-Snell and Nagelkerke pseudo-R^2.

    Parameters
    ----------
    llmod : float
        Maximised log-likelihood of the model under assessment.
    llnull : float
        Maximised log-likelihood of the intercept-only null model.  Must
        be strictly negative -- ``LL0 = 0`` makes (8.70) and (8.72)
        undefined.
    n : int
        Number of observations the model was fitted on.

    Returns
    -------
    RichResult
        Keys: ``mcfadden``, ``coxsnell``, ``nagelkerke``,
        ``coxsnell_max``, ``llmod``, ``llnull``, ``n``.
    """
    llmod = float(llmod)
    llnull = float(llnull)
    n = int(n)
    if not math.isfinite(llmod) or not math.isfinite(llnull):
        raise ValueError("log-likelihoods must be finite")
    if n < 1:
        raise ValueError("n must be at least 1")
    if llnull >= 0.0:
        raise ValueError("llnull must be strictly negative for (8.70) and (8.72) to be defined")
    mcf = 1.0 - llmod / llnull
    cs = 1.0 - math.exp(2.0 * (llnull - llmod) / n)
    csmax = 1.0 - math.exp(2.0 * llnull / n)
    nag = cs / csmax
    return RichResult(
        title="Pseudo coefficients of determination (Hedderich eqs. 8.70-8.72)",
        summary_lines=[
            ("McFadden R^2", mcf),
            ("Cox-Snell R^2", cs),
            ("Nagelkerke R^2", nag),
        ],
        payload={
            "mcfadden": mcf,
            "coxsnell": cs,
            "nagelkerke": nag,
            "coxsnell_max": csmax,
            "llmod": llmod,
            "llnull": llnull,
            "n": n,
        },
    )


def cheatsheet() -> str:
    return "pseudor2(llmod, llnull, n): McFadden/Cox-Snell/Nagelkerke R^2 -- Hedderich eqs. (8.70)-(8.72)."
