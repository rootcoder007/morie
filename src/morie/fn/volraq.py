# morie.fn -- function file (rootcoder007/morie)
"""Realised quadratic variation of a sampled path."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["vol_realised_quadratic_var"]


def vol_realised_quadratic_var(x):
    r"""Quadratic variation of a discretely observed path.

    .. math:: [X]_T = \sum_i (X_{t_i} - X_{t_{i-1}})^2,

    computed from *levels* (prices or log-prices), where realised
    variance takes returns -- the two agree when the returns are the
    first differences of x. Also returns the realised quarticity
    :math:`\tfrac{m}{3} \sum r_i^4`, the ingredient the BNS jump test
    and the HAR-Q model both need.

    Parameters
    ----------
    x : array-like, shape (m,)
        Observed path levels.

    Returns
    -------
    RichResult
        keys: ``qv``, ``rq`` (realised quarticity), ``n_increments``,
        ``method``.

    References
    ----------
    Barndorff-Nielsen, O. E. & Shephard, N. (2004). Power and bipower
    variation with stochastic volatility and jumps. *Journal of
    Financial Econometrics*, 2(1), 1-37. (QV and the quarticity
    normalisation)
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size < 3:
        raise ValueError("need at least 3 observations of the path.")
    r = np.diff(x)
    m = r.size
    return RichResult(
        payload={
            "qv": float((r**2).sum()),
            "rq": float(m / 3.0 * (r**4).sum()),
            "n_increments": int(m),
            "method": "Realised quadratic variation + quarticity",
        }
    )


def cheatsheet():
    return "volraq: [X]_T = sum (dX)^2; RQ = (m/3) sum r^4"
