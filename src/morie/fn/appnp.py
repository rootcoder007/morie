# morie.fn -- function file (rootcoder007/morie)
"""Personalised propagation of neural predictions."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["appnp"]


def appnp(A, H, alpha=0.1, K=10, exact=False, softmax=True):
    """Propagate class predictions by personalised PageRank.

    A graph convolution deep enough to reach distant neighbours also
    over-smooths.  PPNP breaks the link between depth and receptive field
    by keeping the prediction network shallow and propagating its output
    with personalised PageRank:

        Ahat = Dtilde^{-1/2} Atilde Dtilde^{-1/2},   Atilde = A + I
        Z    = alpha (I - (1 - alpha) Ahat)^{-1} H                (PPNP)

    APPNP replaces the inverse with K power-iteration steps of the same
    random walk with restart:

        Z^(0)   = H
        Z^(k+1) = (1 - alpha) Ahat Z^(k) + alpha H
        Z^(K)   = softmax( (1 - alpha) Ahat Z^(K-1) + alpha H )

    Parameters
    ----------
    A : array-like, shape (n, n)
        Adjacency matrix without self-loops; symmetrised internally.
    H : array-like, shape (n, c)
        Per-node predictions f_theta(X), one row per node.
    alpha : float
        Teleport (restart) probability in (0, 1].
    K : int
        Power-iteration steps.
    exact : bool
        Use the closed-form PPNP inverse instead of K iterations.
    softmax : bool
        Apply the row-wise softmax of the final line.

    Returns
    -------
    RichResult
        ``Z``, ``alpha``, ``K``, ``exact``, ``n``, ``c``.

    References
    ----------
    Klicpera, J., Bojchevski, A. and Guennemann, S. (2019), "Predict then
    propagate: graph neural networks meet personalized PageRank",
    International Conference on Learning Representations;
    arXiv:1810.05997.  Sect. 3 gives the PPNP closed form
    Z = alpha (I_n - (1-alpha) Ahat)^{-1} H and the APPNP power iteration
    Z^(0) = H, Z^(k+1) = (1-alpha) Ahat Z^(k) + alpha H, with the softmax
    on the last step.  Read from the ar5iv rendering of the arXiv source.
    """
    Am = C.mat(A)
    n = len(Am)
    if len(Am[0]) != n:
        raise ValueError("A must be square")
    Hm = C.mat(H)
    if len(Hm) != n:
        raise ValueError("H must have one row per node")
    c = len(Hm[0])
    alpha = float(alpha)
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must lie in (0, 1]")
    At = [[(Am[i][j] + Am[j][i]) / 2.0 + (1.0 if i == j else 0.0)
           for j in range(n)] for i in range(n)]
    deg = [sum(r) for r in At]
    if any(d <= 0.0 for d in deg):
        raise ValueError("every node must have positive degree after A + I")
    ds = [1.0 / math.sqrt(d) for d in deg]
    Ah = [[ds[i] * At[i][j] * ds[j] for j in range(n)] for i in range(n)]
    if exact:
        Mm = [[(1.0 if i == j else 0.0) - (1.0 - alpha) * Ah[i][j]
               for j in range(n)] for i in range(n)]
        Z = C.solve(Mm, [[alpha * Hm[i][j] for j in range(c)]
                         for i in range(n)])
        steps = 0
    else:
        steps = int(K)
        Z = [row[:] for row in Hm]
        for _ in range(steps):
            Z = [[(1.0 - alpha) * sum(Ah[i][l] * Z[l][j] for l in range(n))
                  + alpha * Hm[i][j] for j in range(c)] for i in range(n)]
    if softmax:
        out = []
        for r in Z:
            m = max(r)
            e = [math.exp(v - m) for v in r]
            s = sum(e)
            out.append([v / s for v in e])
        Z = out
    return RichResult(payload={
        "Z": Z, "alpha": alpha, "K": steps, "exact": bool(exact),
        "n": n, "c": c,
        "method": "APPNP personalised-PageRank propagation (Klicpera et al. 2019)"})


def cheatsheet():
    return "appnp: Personalised propagation of neural predictions."
