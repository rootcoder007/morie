# morie.fn -- slice s03 (rootcoder007/morie)
"""Transfer entropy.

Source consulted: Schreiber, T. (2000).  Measuring information transfer.
*Physical Review Letters* 85(2), 461-464.  His equation (4) is

    T_(Y->X) = sum p(x_(n+1), x_n^(k), y_n^(l))
               log[ p(x_(n+1) | x_n^(k), y_n^(l)) / p(x_(n+1) | x_n^(k)) ]

equivalently H(X_(n+1) | X_n^(k)) - H(X_(n+1) | X_n^(k), Y_n^(l)), which
is the module's own formula line.  The PRL is paywalled; the definition
is quoted in its standard published form.

Transfer entropy is *directed*: T_(Y->X) and T_(X->Y) are different
numbers, and their difference is the net information flow.  Both are
computed, because reporting only one invites the reader to treat it as a
symmetric measure, which it is not.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["transfer_entropy_te"]


def _te(src, dst, lag):
    n = len(dst)
    L = int(lag)
    trip = {}
    for t in range(L, n - 1):
        key = (dst[t + 1], dst[t - L + 1] if L > 1 else dst[t], src[t - L + 1]
               if L > 1 else src[t])
        trip[key] = trip.get(key, 0.0) + 1.0
    tot = 0.0
    for v in trip.values():
        tot += v
    if tot <= 0.0:
        return float("nan")
    p3 = {kk: v / tot for kk, v in trip.items()}
    p2xy = {}
    p2x = {}
    p1 = {}
    for (a, b, c), v in p3.items():
        p2xy[(b, c)] = p2xy.get((b, c), 0.0) + v
        p2x[(a, b)] = p2x.get((a, b), 0.0) + v
        p1[b] = p1.get(b, 0.0) + v
    te = 0.0
    for kk in sorted(p3):
        a, b, c = kk
        num = p3[kk] / p2xy[(b, c)]
        den = p2x[(a, b)] / p1[b]
        if num > 0.0 and den > 0.0:
            te += p3[kk] * math.log(num / den)
    return te


def transfer_entropy_te(x, y, lag=1):
    """T_(X->Y) and T_(Y->X) for two symbolic series.

    Parameters
    ----------
    x, y : array-like
        The two series, of equal length; entries are treated as symbols.
    lag : int
        The history length k = l.

    Returns
    -------
    estimate : T_(X->Y) in nats
    te_xy, te_yx : both directions
    net      : T_(X->Y) - T_(Y->X)
    """
    a = [str(v) for v in x]
    b = [str(v) for v in y]
    txy = _te(a, b, lag)
    tyx = _te(b, a, lag)
    return RichResult(
        title="Transfer entropy",
        summary_lines=[("T(X->Y)", txy), ("T(Y->X)", tyx)],
        payload={
            "estimate": txy,
            "te_xy": txy,
            "te_yx": tyx,
            "net": txy - tyx,
            "bits": txy / math.log(2.0),
            "lag": int(lag),
            "n": len(a),
            "method": "Transfer entropy (Schreiber 2000, eq. 4); directed, so both directions are reported",
        },
    )


def cheatsheet():
    return "transfen: Transfer entropy TE(X->Y)"
