# morie.fn -- tail3 batch (rootcoder007/morie)
"""Pointwise mutual information (the association ratio).

Source consulted: Church, K.W. & Hanks, P. (1990). Word Association Norms,
Mutual Information, and Lexicography.  *Computational Linguistics* 16(1),
22-29 (ACL Anthology J90-1003).  Their association ratio is the pointwise
mutual information

    I(x, y) = log2 ( P(x, y) / (P(x) P(y)) )

with P(x) = f(x)/N, P(y) = f(y)/N and P(x, y) = f(x, y)/N estimated by
counting co-occurrences.  Church and Hanks note that widening the window to
``w`` words scales the joint count, which subtracts log2(w - 1) from the
score; the ``window`` argument applies that correction.
"""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["pointwise_mutual_info"]


def pointwise_mutual_info(x, y, window=None):
    """Pointwise mutual information of paired co-occurrences.

    Parameters
    ----------
    x, y : array-like
        Equal-length integer codes of the co-occurring pair members.
    window : int, optional
        Window width used to collect the joint counts.  When given,
        ``log2(window - 1)`` is subtracted, per Church & Hanks (1990) note 1.

    Returns
    -------
    RichResult
        estimate (mutual information, bits), pmimax, pmimin, pmimean, npairs,
        n, method.

    References
    ----------
    Church & Hanks (1990), Computational Linguistics 16(1), 22-29.
    """
    xs = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    ys = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
    n = int(min(xs.size, ys.size))
    fx = {}
    fy = {}
    fxy = {}
    for i in range(n):
        a = float(xs[i])
        b = float(ys[i])
        fx[a] = fx.get(a, 0) + 1
        fy[b] = fy.get(b, 0) + 1
        fxy[(a, b)] = fxy.get((a, b), 0) + 1
    adj = 0.0
    if window is not None and int(window) > 1:
        adj = float(np.log2(float(int(window) - 1)))
    nn = float(n)
    pmis = []
    mi = 0.0
    for key in sorted(fxy):
        pxy = fxy[key] / nn
        px = fx[key[0]] / nn
        py = fy[key[1]] / nn
        v = float(np.log2(pxy / (px * py))) - adj
        pmis.append(v)
        mi += pxy * v
    arr = np.asarray(pmis, dtype=float)
    return RichResult(
        payload={
            "estimate": float(mi),
            "mi_bits": float(mi),
            "pmimax": float(np.max(arr)),
            "pmimin": float(np.min(arr)),
            "pmimean": float(np.mean(arr)),
            "pmi": arr,
            "npairs": int(arr.size),
            "n": n,
            "method": "Pointwise mutual information / association ratio (Church & Hanks 1990)",
        }
    )


# CANONICAL TEST
# >>> # perfectly coupled pair over 2 symbols: PMI = log2(2) = 1 bit for both
# >>> r = pointwise_mutual_info([0, 0, 1, 1], [0, 0, 1, 1])
# >>> assert abs(r["estimate"] - 1.0) < 1e-12


def cheatsheet():
    return "pmiwd(x, y): pointwise mutual information / association ratio."
