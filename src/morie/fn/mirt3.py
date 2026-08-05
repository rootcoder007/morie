# morie.fn -- function file (rootcoder007/morie)
"""3D compensatory multidimensional IRT (alias of :mod:`mirt2`)."""

from . import _tail1core as C
from .mirt2 import mirt_2d_compensatory

__all__ = ["mirt_3d_compensatory", "mirt3dcompensatory"]


def mirt_3d_compensatory(y, theta, a, d, D=1.0):
    """Three-dimensional compensatory MIRT, no guessing parameter.

    This module is an ALIAS.  The response function is implemented once,
    in ``mirt2.mirt_2d_compensatory``; the dimension is set by the
    length of ``a``, not by the module name, so the only thing this
    entry point adds is the M2PL restriction ``c = 0`` and a check that
    exactly three slopes were supplied.

        P(x = 1 | theta) = 1 / (1 + exp[-D (a1 t1 + a2 t2 + a3 t3 + d)])

    Parameters
    ----------
    y : array-like of 0/1, length n
        Observed responses.
    theta : array-like, shape (n, 3)
        Ability vectors.
    a : array-like, length 3
        Item slopes.
    d : float
        Item intercept.
    D : float, default 1
        Metric constant.

    Returns
    -------
    RichResult
        As ``mirt2``: ``estimate``, ``loglik``, ``p``, ``pbar``,
        ``deviance``, ``n``, ``m``.

    References
    ----------
    Chalmers, R. P. (2012), Journal of Statistical Software 48(6), 1-29,
    doi:10.18637/jss.v048.i06, eq. (1) p.3, with gamma = 0.
    """
    if len(C.vec(a)) != 3:
        raise ValueError("mirt3 is the three-dimensional case; a must have "
                         "exactly 3 slopes (use mirt2 for other dimensions)")
    return mirt_2d_compensatory(y, theta, a, d, 0.0, D)


mirt3dcompensatory = mirt_3d_compensatory


def cheatsheet():
    return "mirt3: 3D compensatory multidimensional IRT (alias of mirt2)"
