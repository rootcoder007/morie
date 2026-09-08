# morie.fn -- function file (rootcoder007/morie)
"""Placements of the Y sample among the X order statistics."""

import math

from ._richresult import RichResult

__all__ = ['placement', 'gibbons_placement_def']


def placement(x, y):
    """Placement P_(j) of each ordered Y among the X observations.

    Section 2.11 (book p. 65): P_(j) = m S_m(Y_(j)) counts the X's not
    exceeding Y_(j).  The book's two identities are returned as well:
    rank(Y_(j)) = P_(j) + j, and the block frequencies are the first
    differences r_j = P_(j) - P_(j-1) with P_(0) = 0.  The placement
    total equals the Mann-Whitney count of (X, Y) pairs with X < Y.

    Parameters
    ----------
    x, y : sequence of float
        The two samples, sizes m and n.

    Returns
    -------
    RichResult
        keys ``placements``, ``ranks``, ``blocks``, ``total``,
        ``m``, ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 2.11, p. 65 (Orban and Wolfe,
    1982).
    """
    xs = sorted(float(v) for v in x)
    ys = sorted(float(v) for v in y)
    m = len(xs)
    n = len(ys)
    if m < 1 or n < 1:
        raise ValueError("both samples must be non-empty.")
    plc = []
    for yv in ys:
        c = 0
        for xv in xs:
            if xv <= yv:
                c += 1
            else:
                break
        plc.append(c)
    blocks = [plc[0]] + [plc[k] - plc[k - 1] for k in range(1, n)]
    return RichResult(
        payload={
            "placements": plc,
            "ranks": [plc[k] + (k + 1) for k in range(n)],
            "blocks": blocks,
            "total": int(sum(plc)),
            "m": m,
            "n": n,
            "method": "P_(j) = m S_m(Y_(j)); rank(Y_(j)) = P_(j) + j",
        }
    )


gibbons_placement_def = placement
