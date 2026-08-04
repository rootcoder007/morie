# morie.fn -- function file (rootcoder007/morie)
"""Wasserstein-p between two point clouds."""

import math

from . import _s04core as S
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["wasserstein_p_d"]


def wasserstein_p_d(X_samples, Y_samples, p=2.0):
    """Transport distance between equal-size clouds by optimal assignment.

    Above one dimension there is no sorting shortcut, so the coupling is
    an assignment problem.  With equal-size samples the optimal coupling
    is a permutation, and it is found here by the Hungarian method --
    exact, and deterministic, unlike the entropic approximations that
    depend on a regularisation parameter and a stopping rule.

    Formula: ``W_p^p = min_sigma (1 / n) sum_i ||x_i - y_sigma(i)||^p``.

    Parameters
    ----------
    X_samples, Y_samples : array-like, shape (n, d)
        Equal-size point clouds.
    p : float, default 2.0
        Order.

    Returns
    -------
    RichResult
        ``estimate`` (``W_p``), ``wpp`` (``W_p^p``), ``assignment``,
        ``n``.

    References
    ----------
    Villani, C. (2009).  Optimal Transport: Old and New.  Springer,
    Grundlehren 338, chapter 6 (the Wasserstein distances).  The
    assignment step is Kuhn, H. W. (1955), The Hungarian method for the
    assignment problem, Naval Research Logistics Quarterly 2:83-97.
    """
    A = C.mat(X_samples)
    B = C.mat(Y_samples)
    n = len(A)
    d = len(A[0])
    Cst = [[sum(abs(A[i][k] - B[j][k]) ** 2 for k in range(d)) ** (p / 2.0)
            for j in range(n)] for i in range(n)]
    asg = S.hungarian(Cst)
    tot = sum(Cst[i][asg[i]] for i in range(n)) / n
    return RichResult(payload={
        "estimate": tot ** (1.0 / p), "wpp": tot, "assignment": asg, "n": n,
        "method": "Wasserstein-p by optimal assignment"})


wassersteinpd = wasserstein_p_d


def cheatsheet():
    return "wsdt2d: Wasserstein-p between two point clouds."
