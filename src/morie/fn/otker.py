# morie.fn -- function file (rootcoder007/morie)
"""Kernel-induced ground cost with an entropic transport solve."""

import math

from . import _otcore as ot
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["ot_kernel_emd_approx"]


def _gram(A, B, kernel, gamma):
    if kernel == "linear":
        return [[sum(x[k] * y[k] for k in range(len(x))) for y in B] for x in A]
    if kernel == "gaussian":
        out = []
        for x in A:
            out.append([math.exp(-gamma * sum((x[k] - y[k]) ** 2
                                              for k in range(len(x))))
                        for y in B])
        return out
    raise ValueError("kernel must be 'gaussian' or 'linear'")


def ot_kernel_emd_approx(X, Y, kernel="gaussian", epsilon=0.1, gamma=1.0,
                         max_iter=200):
    """Transport in a feature space, with the cost read off a kernel.

    Once the ground cost is the squared distance in a feature space, it
    never has to be formed there: ``||phi(x) - phi(y)||^2 = k(x,x) +
    k(y,y) - 2 k(x,y)``, so the kernel alone determines the cost, which is
    the ``-2 k(x,y)`` of the usual shorthand plus the two diagonal terms
    that keep it non-negative.  Dropping them would shift every entry by a
    row and a column constant, which leaves the optimal plan alone but
    makes the reported cost meaningless.

    Formula: ``C_ij = k(x_i,x_i) + k(y_j,y_j) - 2 k(x_i,y_j)``, then
    ``T`` by Sinkhorn and ``EMD_approx = <T, C>`` -- Genevay, Peyre &
    Cuturi (2018), Section 3.

    Parameters
    ----------
    X, Y : array-like, shape (n, d), (m, d)
        Two point clouds, given uniform weight.
    kernel : {'gaussian', 'linear'}, default 'gaussian'
        Kernel family.
    epsilon : float, default 0.1
        Entropic strength, positive.
    gamma : float, default 1.0
        Gaussian bandwidth parameter.
    max_iter : int, default 200
        Sinkhorn sweeps.

    Returns
    -------
    RichResult
        ``EMD_approx``, ``C``, ``exact_cost`` (the linear-program optimum
        for the same cost), ``n``, ``m``.

    References
    ----------
    Genevay, A., Peyre, G. and Cuturi, M. (2018).  Learning generative
    models with Sinkhorn divergences.  Proceedings of Machine Learning
    Research 84:1608-1617 (AISTATS).
    """
    A = core.mat(X)
    B = core.mat(Y)
    if len(A[0]) != len(B[0]):
        raise ValueError("point clouds must share a dimension")
    n, m = len(A), len(B)
    Kxy = _gram(A, B, kernel, float(gamma))
    kxx = [_gram([A[i]], [A[i]], kernel, float(gamma))[0][0] for i in range(n)]
    kyy = [_gram([B[j]], [B[j]], kernel, float(gamma))[0][0] for j in range(m)]
    C = [[kxx[i] + kyy[j] - 2.0 * Kxy[i][j] for j in range(m)]
         for i in range(n)]
    a = [1.0 / n] * n
    b = [1.0 / m] * m
    T, _, _ = ot.sinkhorn(a, b, C, float(epsilon), max_iter)
    _, exact = ot.emd(a, b, C)
    return RichResult(payload={
        "EMD_approx": ot.frob(T, C), "C": C, "exact_cost": exact,
        "n": n, "m": m,
        "method": "Kernel-induced transport cost"})


def cheatsheet():
    return "otker: entropic transport under a kernel-induced ground cost"
