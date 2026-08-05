# morie.fn -- function file (rootcoder007/morie)
"""Naive gross treatment-effect bound."""

from . import _bndcore as B

from ._richresult import RichResult

__all__ = ["bound_naive_gross"]


def bound_naive_gross(y, D):
    """Naive gross bound on the average treatment effect.

    The treated mean is taken at face value as ``E[y(1)]`` and nothing at
    all is assumed about ``E[y(0)]`` beyond the observed support, so the
    counterfactual mean is placed at each end of that support in turn.
    The width of the resulting interval is exactly the range of ``y``,
    whatever the data look like -- which is the point of the exercise: the
    bound is reported to show how little the data alone deliver.

    Formula: ``[E(y | D = 1) - y_1, E(y | D = 1) - y_0]`` with
    ``y_0 = min y`` and ``y_1 = max y``.

    Parameters
    ----------
    y : array-like
        Observed outcome.
    D : array-like
        Binary treatment indicator, coded 0/1.

    Returns
    -------
    RichResult
        ``lower``, ``upper``, ``width``, ``estimate`` (interval midpoint),
        ``p_treated``, ``n``.

    References
    ----------
    Manski, C. F. (1990).  Nonparametric bounds on treatment effects.
    American Economic Review Papers and Proceedings 80(2), 319-323.
    The decomposition is restated as equation (2.11) of Molinari, F.
    (2021), Microeconometrics with partial identification, Handbook of
    Econometrics 7A, 355-486 (arXiv:2004.11751 p. 17), which is the copy
    used here.
    """
    yv, dv = B.yd(y, D, "bound_naive_gross")
    p1, m1, p0, m0 = B.cellmeans(yv, dv)
    if p1 <= 0.0:
        raise ValueError("bound_naive_gross: no treated unit")
    y0, y1 = B.support(yv)
    lo = m1 - y1
    hi = m1 - y0
    return RichResult(payload={
        "lower": lo, "upper": hi, "width": hi - lo,
        "estimate": 0.5 * (lo + hi), "p_treated": p1, "n": len(yv),
        "method": "Naive gross treatment-effect bound"})


def cheatsheet():
    return "bndnvg: Naive gross treatment-effect bound"
