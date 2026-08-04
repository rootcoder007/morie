# morie.fn -- function file (rootcoder007/morie)
"""Freeman-Tukey double arcsine transform for proportions."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["ma_freeman_tukey"]


def ma_freeman_tukey(x, n):
    """Variance-stabilising double arcsine transform of a proportion.

    Meta-analysis of proportions has to weight studies, and weights want
    a variance that does not depend on the thing being estimated.  A
    single arcsine nearly does that but blows up at zero and one; the
    double arcsine averages two neighbouring single arcsines so a study
    with no events still contributes, which is exactly the case that
    matters in rare-event synthesis.  The variance depends only on
    ``n``, which is what makes the weights fixed rather than estimated.

    Formula: ``FT = arcsin sqrt(x / (n + 1)) + arcsin sqrt((x + 1) / (n + 1))``
    with ``Var(FT) = 1 / (n + 1/2)``.

    Parameters
    ----------
    x : array-like
        Event counts.
    n : array-like
        Sample sizes, elementwise against ``x``.

    Returns
    -------
    RichResult
        ``ft`` (transformed values), ``var``, ``se``, ``k``.

    References
    ----------
    Freeman, M. F. & Tukey, J. W. (1950).  Transformations related to
    the angular and the square root.  Annals of Mathematical Statistics
    21:607-611.  The ``1 / (n + 1/2)`` variance is Miller, J. J. (1978),
    The inverse of the Freeman-Tukey double arcsine transformation,
    The American Statistician 32:138.
    """
    xv = C.vec(x)
    nv = C.vec(n)
    if len(nv) == 1:
        nv = nv * len(xv)
    ft, var = [], []
    for xi, ni in zip(xv, nv):
        ft.append(math.asin(math.sqrt(xi / (ni + 1.0)))
                  + math.asin(math.sqrt((xi + 1.0) / (ni + 1.0))))
        var.append(1.0 / (ni + 0.5))
    return RichResult(payload={
        "ft": ft, "var": var, "se": [math.sqrt(v) for v in var], "k": len(ft),
        "method": "Freeman-Tukey double arcsine transform"})


mafreemantukey = ma_freeman_tukey


def cheatsheet():
    return "mafrt: Freeman-Tukey double arcsine transform for proportions."
