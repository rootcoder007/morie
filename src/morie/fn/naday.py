# morie.fn -- function file (rootcoder007/morie)
"""Nadaraya-Watson kernel regression (alias of :mod:`hrzk2`)."""

from .hrzk2 import horowitz_kernel_regression

__all__ = ["nadaraya_watson", "nadarayawatson"]


def nadaraya_watson(x, y, h=None, grid=None):
    """Nadaraya-Watson kernel regression estimate.

    This module is an ALIAS.  The estimator is implemented once, in
    ``hrzk2.horowitz_kernel_regression``; this entry point supplies the
    classical argument spelling (bandwidth ``h``) and delegates.  No
    second copy of the arithmetic exists.

        m_hat(x) = sum_i K_h(x - X_i) Y_i / sum_i K_h(x - X_i)

    with a Gaussian kernel and, when ``h`` is omitted, Silverman's
    rule-of-thumb bandwidth ``1.06 sigma n^(-1/5)``.

    Parameters
    ----------
    x, y : array-like
        Univariate regressor and response.
    h : float, optional
        Bandwidth.  Default: Silverman's rule of thumb.
    grid : array-like, optional
        Evaluation points; default the sample ``x`` itself.

    Returns
    -------
    RichResult
        ``estimate``, ``se``, ``bandwidth``, ``n``, ``method``.

    References
    ----------
    Nadaraya, E. A. (1964), "On estimating regression", Theory of
    Probability and its Applications 9(1), 141-142,
    doi:10.1137/1109020.  Watson, G. S. (1964), "Smooth regression
    analysis", Sankhya A 26(4), 359-372.
    """
    return horowitz_kernel_regression(x, y, bandwidth=h, grid=grid)


nadarayawatson = nadaraya_watson


def cheatsheet():
    return "naday: Nadaraya-Watson kernel regression (alias of hrzk2)"
