# morie.fn -- function file (rootcoder007/morie)
"""Minibatch optimal transport loss."""

from . import _otcore as ot
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["ot_minibatch_loss"]


def ot_minibatch_loss(X, Y, batch_size, n_batches, epsilon):
    """Average the transport cost over subsamples instead of solving once.

    Full transport is cubic in the sample size and its gradient touches
    every point, which is unusable inside a training loop.  Averaging over
    minibatches costs a constant per step, but it is a genuinely different
    functional: the minibatch loss is not zero between a measure and
    itself, and it does not vanish only at equality.  Both facts are
    reported, so the bias is visible rather than assumed away.

    Formula: ``(1/M) sum_m OT_eps(X_m, Y_m)`` over ``M`` batches --
    Fatras et al. (2020) eq. (3).  The batches here are consecutive
    slices, taken cyclically, so the estimate is deterministic.

    Parameters
    ----------
    X, Y : array-like, shape (n, d), (m, d)
        Two point clouds.
    batch_size : int
        Points per batch.
    n_batches : int
        Number of batches.
    epsilon : float
        Entropic strength, positive.

    Returns
    -------
    RichResult
        ``loss``, ``per_batch``, ``batch_size``, ``n_batches``, ``n``,
        ``m``, ``d``.

    References
    ----------
    Fatras, K., Zine, Y., Flamary, R., Gribonval, R. and Courty, N.
    (2020).  Learning with minibatch Wasserstein: asymptotic and gradient
    properties.  Proceedings of Machine Learning Research 108:2131-2141
    (AISTATS).
    """
    A = core.mat(X)
    B = core.mat(Y)
    n, m = len(A), len(B)
    d = len(A[0])
    if len(B[0]) != d:
        raise ValueError("point clouds must share a dimension")
    bs = int(batch_size)
    M = int(n_batches)
    if bs < 1 or M < 1:
        raise ValueError("batch_size and n_batches must be positive")
    if bs > n or bs > m:
        raise ValueError("batch_size exceeds a cloud")
    eps = float(epsilon)
    u = [1.0 / bs] * bs
    per = []
    for b in range(M):
        xi = [A[(b * bs + t) % n] for t in range(bs)]
        yi = [B[(b * bs + t) % m] for t in range(bs)]
        C = ot.costmat(xi, yi, 2)
        T, _, _ = ot.sinkhorn(u, u, C, eps, 200)
        per.append(ot.frob(T, C))
    return RichResult(payload={
        "loss": sum(per) / M, "per_batch": per, "batch_size": bs,
        "n_batches": M, "n": n, "m": m, "d": d,
        "method": "Minibatch optimal transport loss"})


def cheatsheet():
    return "otmm: minibatch entropic optimal-transport loss"
