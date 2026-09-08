# morie.fn -- slice s03 (rootcoder007/morie)
"""The information bottleneck.

Source consulted (FETCHED): Tishby, N., Pereira, F. C. and Bialek, W.
(1999).  The information bottleneck method.  *Allerton* 37, 368-377
(physics/0004057).  The Lagrangian is

    L = I(X; T) - beta I(T; Y)

and the paper's self-consistent solution, printed there verbatim, is

    p(t | x) = p(t) / Z(x, beta)
               * exp[ -beta sum_y p(y | x) log( p(y | x) / p(y | t) ) ]
    p(t)     = sum_x p(x) p(t | x)
    p(y | t) = (1 / p(t)) sum_x p(y | x) p(t | x) p(x)

iterated to a fixed point.  Note the exponent is the Kullback-Leibler
divergence between p(y | x) and p(y | t) -- the *relevant* distortion,
which is what distinguishes the information bottleneck from ordinary
rate-distortion, where d is given from outside.

DETERMINISM.  The paper's iteration needs an initial p(t | x); a
low-discrepancy deterministic initialisation is used, not a random one.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["information_bottleneck"]


def information_bottleneck(X, Y=None, beta=5.0, T=2, iters=500, tol=1e-14,
                           pxy=None):
    """Iterate the information-bottleneck fixed point.

    Parameters
    ----------
    X, Y : array-like
        Paired discrete observations, used to form the empirical joint.
    beta : float
        The trade-off parameter.
    T : int
        Size of the bottleneck alphabet.
    pxy : 2-D array-like, optional
        The joint p(x, y) directly, instead of X and Y.

    Returns
    -------
    estimate : the Lagrangian I(X;T) - beta I(T;Y)
    ixt, ity : the two mutual informations, in nats
    p_t_x    : the bottleneck encoder
    """
    if pxy is not None:
        J = k.mat(pxy)
    else:
        a = [str(v) for v in X]
        b = [str(v) for v in Y]
        la = sorted(set(a))
        lb = sorted(set(b))
        J = [[0.0] * len(lb) for _ in range(len(la))]
        for i in range(len(a)):
            J[la.index(a[i])][lb.index(b[i])] += 1.0 / len(a)
    n = len(J)
    m = len(J[0])
    px = [0.0] * n
    for i in range(n):
        s = 0.0
        for j in range(m):
            s += J[i][j]
        px[i] = s
    pygx = [[J[i][j] / px[i] if px[i] > 0.0 else 0.0 for j in range(m)]
            for i in range(n)]
    Tn = int(T)
    Q = [[0.0] * Tn for _ in range(n)]
    for i in range(n):
        row = [0.5 + k.vdc(i * Tn + t, 2) for t in range(Tn)]
        s = 0.0
        for v in row:
            s += v
        Q[i] = [v / s for v in row]
    pt = [0.0] * Tn
    pygt = [[0.0] * m for _ in range(Tn)]
    for _ in range(iters):
        for t in range(Tn):
            s = 0.0
            for i in range(n):
                s += px[i] * Q[i][t]
            pt[t] = s
        for t in range(Tn):
            for j in range(m):
                s = 0.0
                for i in range(n):
                    s += pygx[i][j] * Q[i][t] * px[i]
                pygt[t][j] = s / pt[t] if pt[t] > 0.0 else 0.0
        delta = 0.0
        for i in range(n):
            lw = []
            for t in range(Tn):
                kl = 0.0
                for j in range(m):
                    if pygx[i][j] > 0.0 and pygt[t][j] > 0.0:
                        kl += pygx[i][j] * math.log(pygx[i][j] / pygt[t][j])
                    elif pygx[i][j] > 0.0:
                        kl += pygx[i][j] * 700.0
                lw.append(math.log(pt[t] if pt[t] > 1e-300 else 1e-300)
                          - float(beta) * kl)
            z = k.logsumexp(lw)
            for t in range(Tn):
                nv = math.exp(lw[t] - z)
                delta += abs(nv - Q[i][t])
                Q[i][t] = nv
        if delta < tol:
            break
    ixt = 0.0
    for i in range(n):
        for t in range(Tn):
            if Q[i][t] > 0.0 and pt[t] > 0.0:
                ixt += px[i] * Q[i][t] * math.log(Q[i][t] / pt[t])
    py = [0.0] * m
    for j in range(m):
        s = 0.0
        for i in range(n):
            s += J[i][j]
        py[j] = s
    ity = 0.0
    for t in range(Tn):
        for j in range(m):
            if pygt[t][j] > 0.0 and py[j] > 0.0:
                ity += pt[t] * pygt[t][j] * math.log(pygt[t][j] / py[j])
    return RichResult(
        title="Information bottleneck",
        summary_lines=[("I(X;T)", ixt), ("I(T;Y)", ity)],
        payload={
            "estimate": ixt - float(beta) * ity,
            "lagrangian": ixt - float(beta) * ity,
            "ixt": ixt,
            "ity": ity,
            "p_t_x": Q,
            "p_t": pt,
            "beta": float(beta),
            "method": "Information bottleneck fixed point (Tishby, Pereira and Bialek 1999)",
        },
    )


def cheatsheet():
    return "infbtl: Information bottleneck Lagrangian"
