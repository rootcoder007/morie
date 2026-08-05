# morie.fn -- function file (rootcoder007/morie)
"""Mean-sigma equating coefficients."""

from . import _s03core as core
from ._richresult import RichResult

__all__ = ["equating_mean_sigma"]


def equating_mean_sigma(y, b_R, b_F, ddof=1):
    """
    Mean-sigma equating coefficients

    Formula: the scale linking theta_R = A theta_F + B implies
    b_R = A b_F + B for the common items, so matching the first two
    moments of the difficulties gives

        A = sd(b_R) / sd(b_F)
        B = mean(b_R) - A mean(b_F)

    DOCSTRING ERRATUM: the generated stub printed A = sd(b_F)/sd(b_R) and
    B = mean(b_F) - A mean(b_R), i.e. the transformation in the opposite
    direction from its own sibling module eqmm.  Marco's method places
    the NEW form on the REFERENCE metric, which is the orientation
    implemented here; the anchor recovers a known (A, B) applied to
    b_F exactly, which the reversed reading cannot do.

    Parameters
    ----------
    y : array-like
        Scores on the Form F metric to be placed on the Form R metric.
    b_R : array-like
        Common-item difficulties on the Form R (reference) metric.
    b_F : array-like
        The same items' difficulties on the Form F (new form) metric.
    ddof : int
        Denominator correction for the standard deviations; 1 gives the
        sample sd, 0 the population sd.  A cancels in the ratio only when
        both use the same ddof, so it is a single argument.

    Returns
    -------
    result : dict
        Keys: estimate (A), A, B, equated, n_items, n, method.

    References
    ----------
    Marco (1977), Journal of Educational Measurement 14(2):139-160,
    doi:10.1111/j.1745-3984.1977.tb00033.x.
    """
    bR = [float(v) for v in b_R]
    bF = [float(v) for v in b_F]
    k = len(bR)
    if k < 2:
        raise ValueError("need at least two common items")
    if len(bF) != k:
        raise ValueError("b_R and b_F must have the same length")
    ddof = int(ddof)
    if ddof not in (0, 1):
        raise ValueError("ddof must be 0 or 1")
    sF = core.sd(bF, ddof)
    if sF <= 0.0:
        raise ValueError("b_F has zero spread; mean-sigma is undefined")
    A = core.sd(bR, ddof) / sF
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
        "method": "Mean-sigma equating coefficients",
    })


def cheatsheet():
    return "eqms: Mean-sigma equating coefficients"


# compact alias per ledger/NAMING.md
equatingmeansigma = equating_mean_sigma
