# morie.fn -- function file (rootcoder007/morie)
"""One-directional robust LM error (Bera-Yoon)."""

from ._rrng_core import pnorm
from .lmdiag import _rs_core


def lmrerr2(y, X, W):
    r"""One-directional (one-sided) Bera-Yoon robust LM test for positive spatial error dependence.

    The Bera and Yoon (1993) adjusted score for lambda in the presence of
    a local spatial lag, standardised: z = (d_err - T d_lag / nJ) /
    sqrt(T (1 - T / nJ)) (notation of :func:`morie.fn.lmdiag.lmdiag`), is
    asymptotically N(0, 1); z^2 is adjRSerr. The one-directional test
    of lambda > 0 uses the upper normal tail of z (Anselin, Bera,
    Florax and Yoon 1996, sec. 3).

    References
    ----------
    Bera, A. K. and Yoon, M. J. (1993). Specification testing with locally
    misspecified alternatives. *Econometric Theory* 9, 649-658.
    Anselin, L., Bera, A. K., Florax, R. and Yoon, M. J. (1996). Simple
    diagnostic tests for spatial dependence. *Regional Science and Urban
    Economics* 26, 77-104.

    Examples
    --------
    >>> W = [[0, 1, 0, 0, 0], [0.5, 0, 0.5, 0, 0], [0, 0.5, 0, 0.5, 0], [0, 0, 0.5, 0, 0.5], [0, 0, 0, 1, 0]]
    >>> r = lmrerr2([1.0, 2.5, 2.0, 4.5, 4.0], [[0.0], [1.0], [2.0], [3.0], [4.0]], W)
    >>> round(r.statistic, 10)
    -1.0608340907
    """
    from ._containers import SpatialResult

    c = _rs_core(y, X, W)
    z = c["adjRSerr_z"]
    return SpatialResult(
        name="lmrerr2", statistic=z, p_value=float(pnorm(z, lower_tail=False)), extra={"adjRSerr": c["adjRSerr"]}
    )


lmrerr2_fn = lmrerr2


def cheatsheet() -> str:
    return "lmrerr2(y, X, W) -> one-sided Bera-Yoon robust LM error z (z^2 = adjRSerr)."
