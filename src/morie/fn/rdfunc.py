# morie.fn -- slice s03 (rootcoder007/morie)
"""Rate-distortion function by the Blahut-Arimoto algorithm.

Sources consulted: Blahut, R. (1972).  Computation of channel capacity
and rate-distortion functions.  *IEEE Transactions on Information
Theory* 18(4), 460-473; Arimoto, S. (1972), *ibid.* 18(1), 14-20; and
Tishby, N., Pereira, F. C. and Bialek, W. (1999).  The information
bottleneck method.  *Allerton* 37, 368-377 (physics/0004057 -- FETCHED),
whose section 2 reproduces the fixed point verbatim:

    p(xtilde | x) = p(xtilde) / Z(x, beta) * exp[ -beta d(x, xtilde) ]

alternated with p(xtilde) = sum_x p(x) p(xtilde | x).  The Lagrangian
being minimised, also printed there, is

    F = I(X; Xtilde) + beta <d(x, xtilde)>

so beta traces out the rate-distortion curve: R(D) is the convex hull of
(distortion, rate) as beta varies, and -1/beta is the slope of R(D) at
that point.  The 1972 papers are paywalled; the fixed point is quoted
from the fetched 1999 source.

The requested distortion D is met by bisecting on beta, so the returned
point sits on the curve at the requested distortion rather than at an
arbitrary beta.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["rate_distortion"]


def _ba(px, D_mat, beta, iters=500, tol=1e-14):
    n = len(px)
    m = len(D_mat[0])
    q = [1.0 / m] * m
    Q = [[0.0] * m for _ in range(n)]
    for _ in range(iters):
        for i in range(n):
            lw = [math.log(q[j] if q[j] > 1e-300 else 1e-300)
                  - beta * D_mat[i][j] for j in range(m)]
            z = k.logsumexp(lw)
            for j in range(m):
                Q[i][j] = math.exp(lw[j] - z)
        nq = [0.0] * m
        for j in range(m):
            s = 0.0
            for i in range(n):
                s += px[i] * Q[i][j]
            nq[j] = s
        delta = 0.0
        for j in range(m):
            delta += abs(nq[j] - q[j])
        q = nq
        if delta < tol:
            break
    R = 0.0
    dist = 0.0
    for i in range(n):
        for j in range(m):
            if Q[i][j] > 0.0 and q[j] > 0.0:
                R += px[i] * Q[i][j] * math.log(Q[i][j] / q[j])
            dist += px[i] * Q[i][j] * D_mat[i][j]
    return R, dist, q, Q


def rate_distortion(px, distortion=None, D=0.1, beta_hi=1e4, iters=500):
    """R(D) at a target distortion, by bisecting beta.

    Parameters
    ----------
    px : array-like
        Source distribution.
    distortion : 2-D array-like
        d(x, xhat); Hamming by default.
    D : float
        Target distortion.
    beta_hi : float
        Upper end of the beta bracket.

    Returns
    -------
    estimate : R(D) in nats
    rate, distortion_achieved, beta
    q        : the reproduction distribution
    """
    p = k.vec(px)
    tot = 0.0
    for v in p:
        tot += v
    p = [v / tot for v in p]
    n = len(p)
    if distortion is None:
        Dm = [[0.0 if i == j else 1.0 for j in range(n)] for i in range(n)]
    else:
        Dm = k.mat(distortion)
    lo = 0.0
    hi = float(beta_hi)
    R = d0 = 0.0
    q = []
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        R, d0, q, _Q = _ba(p, Dm, mid, iters)
        if d0 > float(D):
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12 * max(1.0, hi):
            break
    beta = 0.5 * (lo + hi)
    R, d0, q, _Q = _ba(p, Dm, beta, iters)
    return RichResult(
        title="Rate-distortion function",
        summary_lines=[("R(D) nats", R), ("D achieved", d0)],
        payload={
            "estimate": R,
            "rate": R,
            "bits": R / math.log(2.0),
            "distortion_achieved": d0,
            "beta": beta,
            "slope": -1.0 / beta if beta > 0.0 else float("nan"),
            "q": q,
            "method": "Blahut-Arimoto rate-distortion, beta bisected to the target distortion",
        },
    )


def cheatsheet():
    return "rdfunc: Rate-distortion function R(D)"


ratedistortion = rate_distortion
