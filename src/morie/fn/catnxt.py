# morie.fn -- function file (rootcoder007/morie)
"""Next-item selection for computerized adaptive testing."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["catnext", "cat_next_item"]


def catnext(items, theta, administered=None, exposure=None, D=1.0):
    """Pick the next item: maximum Fisher information at the current theta.

    Maximum-information selection is optimal for measurement and
    terrible for item security -- it keeps choosing the same few
    high-discrimination items for everyone, so an unconstrained bank
    is exposed within a few hundred examinees.  ``exposure`` applies
    the Sympson-Hetter style multiplier that trades a little
    information for control, and ``information`` is returned for every
    candidate so the cost of that trade is visible.

    ``administered`` is a list of ONE-BASED item indices already used;
    they are excluded.  Ties in information break on the lowest index,
    so the two language arms pick the same item.

    Formula: e = exp(D a (theta - b));  P = c + (d - c) e/(1 + e);
             dP = D a e (d - c)/(1 + e)^2;
             I_j(theta) = dP^2 / (P (1 - P));
             choose argmax_j exposure_j I_j(theta) over unused j

    Parameters
    ----------
    items : array-like, shape (J, 4)
        Item parameters (a, b, c, d).
    theta : float
        Current ability estimate.
    administered : sequence of int, optional
        One-based indices already administered.
    exposure : array-like, optional
        Per-item multiplier in [0, 1] (default: all 1).
    D : float
        Scaling constant.

    Returns
    -------
    RichResult
        ``next_item`` (one-based), ``information`` (all items),
        ``weighted`` (after exposure control), ``max_information``,
        ``n_available``, ``J``.

    References
    ----------
    Item response function, derivative and information verified
    against the reference implementation in the CRAN package ``catR``
    3.17 (Magis & Raiche), functions ``Pi`` and ``Ii``.  ``catR``
    implements the procedures of van der Linden & Glas (eds.),
    Elements of Adaptive Testing (2010), which this row cites; that
    volume was NOT obtainable, so the package source is used as the
    reference implementation.  Maximum-information selection is
    Birnbaum (1968), in Lord & Novick, Statistical Theories of Mental
    Test Scores; the exposure-control multiplier is the idea of
    Sympson & Hetter (1985), Proceedings of the 27th Annual Meeting of
    the Military Testing Association, 973-977, applied here in its
    simplest multiplicative form.
    """
    It = C.mat(items)
    J = len(It)
    if J < 1:
        raise ValueError("the item bank must be non-empty")
    if any(len(r) != 4 for r in It):
        raise ValueError("item rows must be (a, b, c, d)")
    theta = float(theta)
    D = float(D)
    used = set()
    if administered is not None:
        for v in C.vec(administered):
            i = int(v)
            if not 1 <= i <= J:
                raise ValueError("administered indices must lie in 1..J")
            used.add(i)
    if exposure is None:
        ex = [1.0] * J
    else:
        ex = C.vec(exposure)
        if len(ex) != J:
            raise ValueError("exposure must have one entry per item")
        if any(v < 0.0 or v > 1.0 for v in ex):
            raise ValueError("exposure multipliers must lie in [0, 1]")
    info = []
    for a, b, c, d in It:
        e = math.exp(D * a * (theta - b))
        p = c + (d - c) * e / (1.0 + e)
        p = min(1.0 - 1e-10, max(1e-10, p))
        dp = D * a * e * (d - c) / (1.0 + e) ** 2
        info.append(dp * dp / (p * (1.0 - p)))
    wt = [ex[j] * info[j] for j in range(J)]
    avail = [j for j in range(J) if (j + 1) not in used]
    if not avail:
        raise ValueError("every item has been administered")
    best = max(avail, key=lambda j: (wt[j], -j))
    return RichResult(payload={
        "next_item": float(best + 1), "information": info,
        "weighted": wt, "max_information": wt[best],
        "n_available": float(len(avail)), "J": float(J),
        "method": "Maximum-information item selection with exposure control"})


cat_next_item = catnext


def cheatsheet():
    return "catnxt: argmax_j exposure_j I_j(theta) over unadministered items"
