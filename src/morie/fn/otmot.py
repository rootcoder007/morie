# morie.fn -- function file (rootcoder007/morie)
"""Multimarginal entropic optimal transport."""

import math

from . import _otcore as ot
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["ot_multimarginal_iter"]


def ot_multimarginal_iter(margins, C_tensor, epsilon, max_iter=200):
    """Couple more than two histograms at once.

    Barycenters, Wasserstein-Bures interpolation and incompressible fluid
    paths are all the same object seen from different angles: a single
    coupling of ``S`` measures, not a chain of pairwise ones.  Pairwise
    plans cannot be glued into a consistent joint law in general, so the
    multimarginal problem has to be solved as one problem.  Entropic
    smoothing keeps the scalings one-dimensional, which is the only reason
    it is tractable at all.

    Formula: ``min_P <C, P> - eps H(P)`` over tensors whose ``S``
    marginals are the given histograms; solved by
    ``u_s <- u_s a_s / marg_s(P)`` -- Peyre & Cuturi (2019) eq. (10.1)
    -(10.2), p. 159, read from the rendered page; Benamou et al. (2015)
    Section 5.

    Parameters
    ----------
    margins : sequence of S arrays
        The prescribed marginals.
    C_tensor : array-like
        Cost tensor, flattened row-major (last index varying fastest),
        of length ``prod_s len(margins[s])``.
    epsilon : float
        Entropic strength, positive.
    max_iter : int, default 200
        Sweeps over the marginals.

    Returns
    -------
    RichResult
        ``T`` (the flattened plan), ``cost``, ``mass``, ``marg_err``,
        ``dims``, ``S``, ``iters``.

    References
    ----------
    Benamou, J.-D., Carlier, G., Cuturi, M., Nenna, L. and Peyre, G.
    (2015).  SIAM Journal on Scientific Computing 37(2):A1111-A1138.
    doi:10.1137/141000439.
    """
    ms = [ot.hist(a) for a in margins]
    S = len(ms)
    if S < 2:
        raise ValueError("a multimarginal problem needs at least two margins")
    dims = [len(a) for a in ms]
    total = 1
    for dnum in dims:
        total *= dnum
    C = [float(t) for t in core.vec(C_tensor)]
    if len(C) != total:
        raise ValueError("the cost tensor does not match the marginal sizes")
    eps = float(epsilon)
    if eps <= 0.0:
        raise ValueError("epsilon must be positive")
    strides = [1] * S
    for s in range(S - 2, -1, -1):
        strides[s] = strides[s + 1] * dims[s + 1]
    idx = [[(t // strides[s]) % dims[s] for s in range(S)]
           for t in range(total)]
    K = [math.exp(-c / eps) for c in C]
    u = [[1.0] * dims[s] for s in range(S)]
    it = int(max_iter)
    P = list(K)
    for _ in range(it):
        for s in range(S):
            P = [K[t] * _prod(u, idx[t], S) for t in range(total)]
            marg = [0.0] * dims[s]
            for t in range(total):
                marg[idx[t][s]] += P[t]
            for i in range(dims[s]):
                if marg[i] > 0.0:
                    u[s][i] *= ms[s][i] / marg[i]
    P = [K[t] * _prod(u, idx[t], S) for t in range(total)]
    err = 0.0
    for s in range(S):
        marg = [0.0] * dims[s]
        for t in range(total):
            marg[idx[t][s]] += P[t]
        for i in range(dims[s]):
            e = abs(marg[i] - ms[s][i])
            if e > err:
                err = e
    return RichResult(payload={
        "T": P, "cost": sum(P[t] * C[t] for t in range(total)),
        "mass": sum(P), "marg_err": err, "dims": dims, "S": S, "iters": it,
        "method": "Multimarginal entropic optimal transport"})


def _prod(u, ix, S):
    v = 1.0
    for s in range(S):
        v *= u[s][ix[s]]
    return v


def cheatsheet():
    return "otmot: multimarginal entropic optimal transport by tensor scaling"
