# morie.fn -- function file (rootcoder007/morie)
"""Viterbi most-likely state path."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["viterbi"]


def viterbi(obs, trans, emit, init=None):
    """Most likely hidden state sequence, in one forward pass.

    The greedy per-step choice is not the best path: a state that looks
    good now can be unreachable from the states that follow.  Viterbi
    fixes that by carrying, for every state, the best path that ends
    there, so the exponential search collapses to ``O(T K^2)``.  Work is
    in log space because a product of a thousand probabilities
    underflows to zero long before the path is decided.

    Formula: ``delta_t(j) = max_i [delta_{t-1}(i) + log A_ij] + log B_j(o_t)``,
    with backpointers to recover the path.

    Parameters
    ----------
    obs : array-like, shape (T,)
        Zero-based observation symbols.
    trans : array-like, shape (K, K)
        Transition probabilities.
    emit : array-like, shape (K, M)
        Emission probabilities.
    init : array-like, optional
        Initial state distribution; uniform by default.

    Returns
    -------
    RichResult
        ``path`` (zero-based states), ``estimate`` (log probability of
        that path), ``T``, ``K``.

    References
    ----------
    Viterbi, A. J. (1967).  Error bounds for convolutional codes and an
    asymptotically optimum decoding algorithm.  IEEE Transactions on
    Information Theory 13:260-269.
    """
    o = [int(round(v)) for v in C.vec(obs)]
    A = C.mat(trans)
    B = C.mat(emit)
    K = len(A)
    T = len(o)
    NEG = -1e300
    def lg(v):
        return math.log(v) if v > 0.0 else NEG
    pi = C.vec(init) if init is not None else [1.0 / K] * K
    delta = [lg(pi[j]) + lg(B[j][o[0]]) for j in range(K)]
    psi = []
    for t in range(1, T):
        nd, np_ = [], []
        for j in range(K):
            best, arg = NEG, 0
            for i in range(K):
                v = delta[i] + lg(A[i][j])
                if v > best:
                    best, arg = v, i
            nd.append(best + lg(B[j][o[t]]))
            np_.append(arg)
        delta, _ = nd, None
        psi.append(np_)
    last = max(range(K), key=lambda j: delta[j])
    path = [last]
    for t in range(T - 2, -1, -1):
        last = psi[t][last]
        path.append(last)
    path.reverse()
    return RichResult(payload={
        "path": path, "estimate": max(delta), "T": T, "K": K,
        "method": "Viterbi most-likely state path"})


def cheatsheet():
    return "viterb: Viterbi most-likely state path."
