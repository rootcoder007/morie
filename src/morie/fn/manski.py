# morie.fn -- function file (rootcoder007/morie)
"""Manski no-assumption bounds on the ATE (forwards to bndest)."""

from .bndest import bound_estimation

__all__ = ["manski", "manski_no_assumption_bounds"]


def manski(y, D, y_min, y_max):
    """Manski (1990) no-assumption (worst-case) bounds on the ATE.

    This module and ``bndest`` document the SAME method: decompose each
    counterfactual mean by the law of total probability and fill the
    unobserved arm with the support endpoints,

        E[Y(t)] in [ E[Y|D=t] P(D=t) + y_min P(D!=t),
                     E[Y|D=t] P(D=t) + y_max P(D!=t) ],

    then difference the arm bounds for the ATE.  Rather than carry a
    second implementation -- which would agree with the first at 1e-9
    forever while doubling the surface -- this function forwards to
    :func:`morie.fn.bndest.bound_estimation` with the argument layout of
    its own stub.  The ATE interval always has width exactly
    ``y_max - y_min`` and therefore always contains zero.

    Parameters
    ----------
    y : array-like
        Observed outcome.
    D : array-like of 0/1
        Treatment indicator.
    y_min, y_max : float
        Logical support of the outcome (the only assumption used).

    Returns
    -------
    RichResult
        ``ate_lower``, ``ate_upper``, ``ate_width``, ``y1_bounds``,
        ``y0_bounds``, ``p_treated``, ``contains_zero``, ``n``.

    References
    ----------
    Manski, C. F. (1990), "Nonparametric Bounds on Treatment Effects",
    American Economic Review Papers and Proceedings 80(2):319-323.
    Molinari, F. (2021), "Microeconometrics with Partial Identification",
    Handbook of Econometrics 7A, eq. (2.11) and p. 18 (ATE differencing);
    local source ~/work/scratch/x000/molinari.pdf (arXiv:2004.11751).
    """
    return bound_estimation(y, None, (y_min, y_max), treatment=D)


# stub-era long name, kept as an alias
manski_no_assumption_bounds = manski


def cheatsheet():
    return "manski: Manski (1990) no-assumption ATE bounds -- forwards to bndest"
