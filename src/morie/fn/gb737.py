# morie.fn -- function file (rootcoder007/morie)
"""Symmetry properties of linear rank statistics -- Theorem 7.3.7."""

import math

from ._richresult import RichResult

__all__ = ['lrankprop', 'gibbons_linrank_properties']


def lrankprop(a, z):
    """The two identities of Theorem 7.3.7, checked on real scores.

    Theorem 7.3.7 (book p. 283).  For T = sum a_i Z_i:

    Property 1: with T' = sum a_i Z_{N-i+1} (the reversed indicator
    vector), T = T' whenever the scores are palindromic,
    a_i = a_{N-i+1}.

    Property 2: with T' = sum a_i (1 - Z_i) (the conjugate, i.e. the
    same scores applied to the other sample), T + T' = sum a_i.

    Both are returned as computed quantities together with the
    corresponding residual, so the identity is verified rather than
    asserted.

    Parameters
    ----------
    a : sequence of float
        Scores a_1, ..., a_N.
    z : sequence of int
        Indicators Z_1, ..., Z_N (1 if the i-th smallest is an X).

    Returns
    -------
    RichResult
        keys ``t``, ``t_reversed``, ``t_conjugate``, ``sum_a``,
        ``palindromic`` (1/0), ``resid1`` (T - T' under property 1),
        ``resid2`` (T + T'' - sum a), ``N``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Theorem 7.3.7, p. 283.
    """
    av = [float(v) for v in a]
    zv = [float(v) for v in z]
    nn = len(av)
    if len(zv) != nn:
        raise ValueError("a and z must have the same length.")
    if nn < 1:
        raise ValueError("a must be non-empty.")
    t = sum(av[i] * zv[i] for i in range(nn))
    trev = sum(av[i] * zv[nn - 1 - i] for i in range(nn))
    tcon = sum(av[i] * (1.0 - zv[i]) for i in range(nn))
    pal = all(abs(av[i] - av[nn - 1 - i]) < 1e-12 for i in range(nn))
    return RichResult(
        payload={
            "t": float(t),
            "t_reversed": float(trev),
            "t_conjugate": float(tcon),
            "sum_a": float(sum(av)),
            "palindromic": int(pal),
            "resid1": float(t - trev),
            "resid2": float(t + tcon - sum(av)),
            "N": int(nn),
            "method": "linear rank statistic properties, Theorem 7.3.7",
        }
    )


gibbons_linrank_properties = lrankprop
