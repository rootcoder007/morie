# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Hidden Markov part-of-speech tagging by the Viterbi algorithm.

Charniak (1993), *Statistical Language Learning*, MIT Press, chapter
3, states the tagging problem as

    argmax_y prod_t P(y_t | y_{t-1}) P(x_t | y_t),

maximised by Viterbi (1967) dynamic programming in O(T |S|^2) rather
than the |S|^T sequences a brute-force search would visit.  Working in
logs keeps the recursion numerically safe for long sentences.  The
tests check the returned path against exhaustive enumeration on a
short sentence, which is the only way to be sure the back-pointers are
right.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["hmm_pos"]

_NEG = float("-inf")


def _lg(p):
    return math.log(p) if p > 0 else _NEG


def hmm_pos(X, tagset, start=None, trans=None, emit=None):
    """Viterbi tag sequence for observed word indices X.

    Parameters
    ----------
    X : array-like of int
        Word indices, 0-based.
    tagset : array-like
        Tag labels; only the count matters here.
    start : length-S initial distribution.
    trans : S x S transition matrix, rows summing to one.
    emit : S x V emission matrix, rows summing to one.
    """
    xs = [int(v) for v in core.vec(X)]
    T = len(xs)
    if T == 0:
        raise ValueError("hmm_pos: X is empty")
    S = len(core.vec(tagset)) if not hasattr(tagset, "__len__") else len(tagset)
    if S < 1:
        raise ValueError("hmm_pos: tagset is empty")
    if start is None or trans is None or emit is None:
        raise ValueError("hmm_pos: start, trans and emit must be supplied")
    pi = core.vec(start)
    A = core.mat(trans)
    B = core.mat(emit)
    if len(pi) != S or len(A) != S or len(B) != S:
        raise ValueError("hmm_pos: start, trans and emit must match the tagset size")
    V = len(B[0])
    for v in xs:
        if v < 0 or v >= V:
            raise ValueError("hmm_pos: observation index out of range")
    delta = [[_NEG] * S for _ in range(T)]
    psi = [[0] * S for _ in range(T)]
    for s in range(S):
        delta[0][s] = _lg(pi[s]) + _lg(B[s][xs[0]])
    for t in range(1, T):
        for s in range(S):
            best = _NEG
            arg = 0
            for r in range(S):
                v = delta[t - 1][r] + _lg(A[r][s])
                if v > best:
                    best = v
                    arg = r
            delta[t][s] = best + _lg(B[s][xs[t]])
            psi[t][s] = arg
    end = 0
    for s in range(S):
        if delta[T - 1][s] > delta[T - 1][end]:
            end = s
    path = [0] * T
    path[T - 1] = end
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1][path[t + 1]]
    return RichResult(
        title="HMM part-of-speech tagging",
        summary_lines=[("length", T), ("tags", S)],
        payload={
            "estimate": delta[T - 1][end],
            "path": [p + 1 for p in path],
            "logprob": delta[T - 1][end],
            "n": T,
            "method": "Viterbi maximisation of prod P(y_t|y_{t-1}) P(x_t|y_t), Charniak (1993) ch. 3",
        },
    )


def cheatsheet():
    return "hmmTag: HMM part-of-speech tagging"


# compact alias per ledger/NAMING.md
hmmpos = hmm_pos
