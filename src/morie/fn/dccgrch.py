# morie.fn -- function file (rootcoder007/morie)
"""Dynamic conditional correlation GARCH -- front-end to the DCC engine."""

from ._richresult import RichResult

__all__ = ["dcc_garch"]


def dcc_garch(X):
    r"""DCC(1,1) GARCH front-end.

    .. math:: R_t = \mathrm{diag}(Q_t)^{-1/2} Q_t \,
              \mathrm{diag}(Q_t)^{-1/2}

    with :math:`Q_t = (1 - a - b)\bar{Q} + a\,z_{t-1}z_{t-1}' +
    b\,Q_{t-1}` (Engle 2002, eqs. 9-11). Delegates to
    :func:`morie.fn.dccmd.dcc_multivariate_garch`, which holds the
    two-step Gaussian MLE; this module exists under the historical
    short name. It replaces a placeholder that computed a Spearman
    correlation of the input with itself (identically 1).

    Parameters
    ----------
    X : array-like, shape (n, k)
        Multivariate return series, n >= 30, k >= 2.

    Returns
    -------
    RichResult
        Same payload as ``dcc_multivariate_garch``: ``a``, ``b``,
        ``unconditional_correlation``, ``conditional_correlation``,
        ``conditional_variance``, ``loglik``, ``n``, ``k``,
        ``method``.

    References
    ----------
    Engle, R. F. (2002). Dynamic conditional correlation. *Journal of
    Business & Economic Statistics*, 20(3), 339-350.
    """
    from .dccmd import dcc_multivariate_garch

    return dcc_multivariate_garch(X)


def cheatsheet():
    return "dccgrch: DCC(1,1) GARCH (front-end to dccmd)"
