# morie.fn -- function file (rootcoder007/morie)
"""Quantization distortion read as an optimal-transport cost."""

from . import _otcore as ot
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["ot_quantization_distortion"]


def ot_quantization_distortion(X, centroids):
    """Distortion of a codebook, computed as transport to its atoms.

    Pollard's formulation of k-means is exactly a transport problem: the
    empirical measure is pushed onto a finitely supported one, and the
    distortion is the transport cost.  Because nearest-point assignment is
    itself an optimal plan for the induced weights, the transport cost and
    the ordinary mean-squared distortion coincide -- so the two are
    computed separately here and both reported, which turns the identity
    into a self-check.

    Formula: ``Distortion = OT(mu, nu_quant)`` with ``nu_quant`` supported
    on the centroids and weighted by the assignment proportions; equal to
    ``(1/n) sum_i min_k ||x_i - c_k||^2`` -- Pollard (1982).

    Parameters
    ----------
    X : array-like, shape (n, d)
        Data points, given equal weight.
    centroids : array-like, shape (K, d)
        Codebook.

    Returns
    -------
    RichResult
        ``dist`` (the transport cost), ``dist_assign`` (the mean-squared
        distortion), ``gap`` (their difference), ``labels``, ``weights``,
        ``n``, ``K``, ``d``.

    References
    ----------
    Pollard, D. (1982).  Quantization and the method of k-means.  IEEE
    Transactions on Information Theory 28(2):199-205.
    doi:10.1109/TIT.1982.1056481.
    """
    A = core.mat(X)
    Cn = core.mat(centroids)
    n, K = len(A), len(Cn)
    d = len(A[0])
    if len(Cn[0]) != d:
        raise ValueError("centroids must live in the same space as the data")
    if n == 0 or K == 0:
        raise ValueError("empty data or codebook")
    C = ot.costmat(A, Cn, 2)
    labels = []
    tot = 0.0
    for i in range(n):
        k = 0
        for j in range(1, K):
            if C[i][j] < C[i][k]:
                k = j
        labels.append(k)
        tot += C[i][k]
    dist_assign = tot / n
    w = [0.0] * K
    for k in labels:
        w[k] += 1.0 / n
    _, cost = ot.emd([1.0 / n] * n, w, C)
    return RichResult(payload={
        "dist": cost, "dist_assign": dist_assign, "gap": abs(cost - dist_assign),
        "labels": labels, "weights": w, "n": n, "K": K, "d": d,
        "method": "Quantization distortion as transport cost"})


def cheatsheet():
    return "otmqd: codebook distortion as an optimal-transport cost"
