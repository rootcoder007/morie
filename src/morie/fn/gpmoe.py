# morie.fn -- function file (rootcoder007/morie)
"""Mixture of Gaussian process experts (Tresp)."""

import math

from . import _s04core as S
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["gp_mixture_of_experts"]


def gp_mixture_of_experts(X, y, X_test, K, ell=1.0, noise=1e-6):
    """Predict from K Gaussian-process experts under a softmax gate.

    One GP over the whole input is both slow and wrong when different
    regions of the input want different smoothness.  Tresp fits several
    GPs and lets a gate decide, per test point, which one to believe.
    The predictive mean is the gate-weighted average of the expert
    means; the predictive variance picks up two pieces, the experts own
    uncertainty and their disagreement, which is why a point midway
    between two confident experts is correctly reported as uncertain.

    Determinism: experts partition the training set into K contiguous
    blocks along the sorted first coordinate, and the gate is a softmax
    of negative squared distance to each block centroid.  No EM, no
    random initialisation, so the answer is a function of the inputs
    alone.

    Formula: ``mu(x) = sum_k pi_k(x) mu_k(x)`` and
    ``var(x) = sum_k pi_k(x) [sigma_k^2(x) + mu_k(x)^2] - mu(x)^2``,
    with ``pi_k(x) = softmax_k(-||x - c_k||^2)``.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Training inputs.
    y : array-like, shape (n,)
        Training targets.
    X_test : array-like, shape (m, p)
        Test inputs.
    K : int
        Number of experts.
    ell : float, default 1.0
        Squared-exponential length scale.
    noise : float, default 1e-6
        Observation noise added to each expert diagonal.

    Returns
    -------
    RichResult
        ``estimate`` (mean over the test points), ``mean``, ``var``,
        ``gate`` (m by K), ``n``, ``K``.

    References
    ----------
    Tresp, V. (2001).  Mixtures of Gaussian processes.  Advances in
    Neural Information Processing Systems 13, 654-660.  Fetched from
    the NeurIPS proceedings; the paper fits M GP experts under a gating
    network and combines them as a mixture, which is the predictive
    moment pair implemented here.
    """
    Xm = C.mat(X)
    Xt = C.mat(X_test)
    yv = C.vec(y)
    n = len(Xm)
    m = len(Xt)
    K = int(K)
    o = S.order([row[0] for row in Xm])
    blocks = [[] for _ in range(K)]
    for pos, idx in enumerate(o):
        blocks[min(pos * K // n, K - 1)].append(idx)
    cent, mu_k, var_k = [], [], []
    for b in blocks:
        b = b if b else [o[0]]
        p = len(Xm[0])
        cent.append([sum(Xm[i][j] for i in b) / len(b) for j in range(p)])
        Xb = [Xm[i] for i in b]
        yb = [yv[i] for i in b]
        Kb = S.rbf(Xb, Xb, ell)
        Ks = S.rbf(Xb, Xt, ell)
        mk, vk = S.gppost(Kb, Ks, [1.0] * m, yb, noise)
        mu_k.append(mk)
        var_k.append(vk)
    gate, mean, var = [], [], []
    for j in range(m):
        d = [-sum((Xt[j][t] - cent[k][t]) ** 2 for t in range(len(Xt[j]))) for k in range(K)]
        pi = S.softmax(d)
        gate.append(pi)
        mj = sum(pi[k] * mu_k[k][j] for k in range(K))
        vj = sum(pi[k] * (var_k[k][j] + mu_k[k][j] ** 2) for k in range(K)) - mj * mj
        mean.append(mj)
        var.append(vj)
    return RichResult(payload={
        "estimate": sum(mean) / m, "mean": mean, "var": var, "gate": gate,
        "n": n, "K": K, "method": "Mixture of GP experts (Tresp)"})


def cheatsheet():
    return "gpmoe: Mixture of Gaussian process experts (Tresp)."
