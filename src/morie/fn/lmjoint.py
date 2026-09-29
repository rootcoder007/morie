# morie.fn -- function file (rootcoder007/morie)
"""Joint LM test for spatial lag and error."""

from .lmdiag import _result, _rs_core


def lmjoint(y, X, W):
    r"""Joint Rao score test of no spatial lag and no spatial error, SARMA = adjRSlag + RSerr (2 df).

    Anselin, Bera, Florax and Yoon (1996) show that the joint LM test of
    rho = lambda = 0 in the SARMA model decomposes as RSerr + adjRSlag
    (equivalently RSlag + adjRSerr); identical to :func:`morie.fn.lmsarma.lmsarma`
    and spdep::lm.RStests(test = "SARMA").

    References
    ----------
    Anselin, L., Bera, A. K., Florax, R. and Yoon, M. J. (1996). Simple
    diagnostic tests for spatial dependence. *Regional Science and Urban
    Economics* 26, 77-104.

    Examples
    --------
    >>> W = [[0, 1, 0, 0, 0], [0.5, 0, 0.5, 0, 0], [0, 0.5, 0, 0.5, 0], [0, 0, 0.5, 0, 0.5], [0, 0, 0, 1, 0]]
    >>> r = lmjoint([1.0, 2.5, 2.0, 4.5, 4.0], [[0.0], [1.0], [2.0], [3.0], [4.0]], W)
    >>> round(r.statistic, 10)
    3.7535691455
    """
    c = _rs_core(y, X, W)
    return _result("lmjoint", c["SARMA"], 2, {"RSlag_plus_adjRSerr": c["RSlag"] + c["adjRSerr"]})


lmjoint_fn = lmjoint


def cheatsheet() -> str:
    return "lmjoint(y, X, W) -> joint LM test rho = lambda = 0, RSerr + adjRSlag (2 df)."
