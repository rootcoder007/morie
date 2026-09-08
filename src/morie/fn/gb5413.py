# morie.fn -- function file (rootcoder007/morie)
"""Zero differences in the sign test: the three standard conventions."""

import math

from ._richresult import RichResult

__all__ = ['signzero', 'gibbons_sign_zeros']


def signzero(x, m0=0.0, method="discard"):
    """Resolve zero differences before applying the sign test.

    Section 5.4.8 (book p. 180) lists the practical conventions.
    ``"discard"`` drops the zeros and reduces N (the book's own
    recommendation), ``"half"`` splits them evenly between the two
    signs (K gains half the zeros, N is kept), and ``"conservative"``
    assigns every zero to the tail that argues against rejection, i.e.
    to the smaller of K and N - K.

    Parameters
    ----------
    x : sequence of float
        Sample or paired differences.
    m0 : float, optional
        Hypothesised median (default 0).
    method : str, optional
        ``"discard"``, ``"half"`` or ``"conservative"``.

    Returns
    -------
    RichResult
        keys ``statistic`` (adjusted K), ``n`` (adjusted N),
        ``nzero``, ``k_raw``, ``n_raw``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 5.4.8, p. 180.
    """
    xs = [float(v) - float(m0) for v in x]
    n_raw = len(xs)
    if n_raw < 1:
        raise ValueError("x must be non-empty.")
    nz = sum(1 for d in xs if d == 0.0)
    kpos = sum(1 for d in xs if d > 0.0)
    kneg = n_raw - nz - kpos
    if method == "discard":
        k, n = float(kpos), n_raw - nz
    elif method == "half":
        k, n = kpos + nz / 2.0, n_raw
    elif method == "conservative":
        k, n = (kpos + nz, n_raw) if kpos < kneg else (float(kpos), n_raw)
    else:
        raise ValueError("method must be discard, half or conservative.")
    return RichResult(
        payload={
            "statistic": float(k),
            "n": int(n),
            "nzero": int(nz),
            "k_raw": int(kpos),
            "n_raw": int(n_raw),
            "method": "sign test zero handling (%s), Sec. 5.4.8" % method,
        }
    )


gibbons_sign_zeros = signzero
