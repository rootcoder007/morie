# morie.fn -- function file (rootcoder007/morie)
"""Average derivative estimator (sample form)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["hrz_average_derivative_hat", "horowitz_improved_ade"]


from .hrzade import hrz_average_derivative


def hrz_average_derivative_hat(X, y, h=None):
    r"""Sample average derivative (Horowitz Sec. 2.6):

    .. math:: \hat\delta = -\frac2n \sum_i \hat f'(X_i)
              \big[Y_i - \hat E(Y|X_i)\big] \Big/
              \hat E[\hat f(X)].

    The explicit estimator behind :mod:`morie.fn.hrzade`. Two details
    carry the root-n property: the density derivative is computed
    LEAVE-ONE-OUT, and the bandwidth must UNDERSMOOTH relative to the
    density-optimal choice so the bias vanishes faster than
    :math:`n^{-1/2}`. A density-optimal bandwidth leaves a bias of the
    same order as the standard error and the confidence interval is
    then centred on the wrong value.

    Parameters
    ----------
    X, y : array-like
        Covariates and response.
    h : float, optional
        Bandwidth; an undersmoothed default otherwise.

    Returns
    -------
    RichResult
        keys: ``delta_hat``, ``se``, ``bandwidth``,
        ``undersmoothed`` (True), ``n``, ``method``.
    References
    ----------
    Horowitz, J. L. *Semiparametric and Nonparametric Methods in
    Econometrics*. Springer. Ch. 2, Sec. 2.6.1-2.6.2 (average-derivative estimators;
    an improved average-derivative estimator).
    """
    from ._horowitz import silverman_bw

    Xa = np.atleast_2d(np.asarray(X, dtype=float))
    ya = np.asarray(y, dtype=float).ravel()
    if Xa.shape[0] != ya.size:
        Xa = Xa.T
    n = Xa.shape[0]
    if h is None:
        # undersmooth: n^{-1/5} * n^{-1/20} shrinks the bias faster
        h = float(silverman_bw(Xa[:, 0]) * n ** (-0.05))
    out = hrz_average_derivative(Xa, ya, h=h)
    return RichResult(payload={"delta_hat": out["delta"], "se": out["se"],
                               "bandwidth": out["bandwidth"],
                               "undersmoothed": True, "n": out["n"],
                               "method": "Sample average derivative; LOO and undersmoothing are required"})


def cheatsheet():
    return "hrzades: a density-optimal bandwidth would bias the CI off-centre"


#: Catalogue alias for :func:`hrz_average_derivative_hat`.
horowitz_improved_ade = hrz_average_derivative_hat
