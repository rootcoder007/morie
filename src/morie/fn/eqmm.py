# morie.fn -- function file (rootcoder007/morie)
"""Mean-mean equating coefficients."""

from . import _s03core as core
from ._richresult import RichResult

__all__ = ["equating_mean_mean"]


def equating_mean_mean(y, a_R, b_R, a_F, b_F):
    """
    Mean-mean equating coefficients

    Formula: the scale linking theta_R = A theta_F + B carries the IRT
    item parameters as b_R = A b_F + B and a_R = a_F / A.  Averaging each
    relation over the common items gives the mean-mean coefficients

        A = mean(a_F) / mean(a_R)
        B = mean(b_R) - A mean(b_F)

    DOCSTRING ERRATUM: the generated stub printed A = mean(a_R)/mean(a_F).
    That is the reciprocal.  Discrimination is inversely proportional to
    the scale factor -- stretching the ability metric by A flattens the
    item characteristic curve by the same factor -- so the ratio must be
    a_F over a_R.  The B formula in the stub was already correct, and the
    two are only mutually consistent with A as written here; the round
    trip in this module's anchor demonstrates it.

    Parameters
    ----------
    y : array-like
        Scores on the Form F metric to be placed on the Form R metric.
    a_R, b_R : array-like
        Common-item discriminations and difficulties on the Form R
        (reference) metric.
    a_F, b_F : array-like
        The same items' parameters on the Form F (new form) metric.

    Returns
    -------
    result : dict
        Keys: estimate (A), A, B, equated, n_items, n, method.

    References
    ----------
    Loyd & Hoover (1980), Journal of Educational Measurement
    17(3):179-193, doi:10.1111/j.1745-3984.1980.tb00825.x.
    """
    aR = [float(v) for v in a_R]
    bR = [float(v) for v in b_R]
    aF = [float(v) for v in a_F]
    bF = [float(v) for v in b_F]
    k = len(aR)
    if k == 0:
        raise ValueError("empty input: no common items")
    if len(bR) != k or len(aF) != k or len(bF) != k:
        raise ValueError("all item-parameter vectors must have the same length")
    if any(v <= 0.0 for v in aR) or any(v <= 0.0 for v in aF):
        raise ValueError("discriminations must be positive")
    mA = core.mean(aR)
    mF = core.mean(aF)
    A = mF / mA
    B = core.mean(bR) - A * core.mean(bF)
    yy = [float(v) for v in ([y] if isinstance(y, (int, float)) else y)]
    eq = [A * v + B for v in yy]
    return RichResult(payload={
        "estimate": A,
        "A": A,
        "B": B,
        "equated": eq,
        "n_items": k,
        "n": len(yy),
        "method": "Mean-mean equating coefficients",
    })


def cheatsheet():
    return "eqmm: Mean-mean equating coefficients"


# compact alias per ledger/NAMING.md
equatingmeanmean = equating_mean_mean
