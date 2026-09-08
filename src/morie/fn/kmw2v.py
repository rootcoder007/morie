# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Word2Vec skip-gram log-likelihood (full softmax)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_word2vec_skipgram"]


def kamath_word2vec_skipgram(center_indices, context_indices, V, U):
    """L = sum over (center, context) pairs of log P(w_o | w_c), with
    P = softmax over the vocabulary of u_o . v_c.

    The FULL softmax, as the spec line writes it: the normaliser runs
    over the whole vocabulary, which is exactly why the implementations
    people actually train use negative sampling or hierarchical
    softmax instead. Reported as a total AND a per-pair mean, since
    comparing corpora by a total log-likelihood compares their
    lengths.

    Reference: the worklist cites Kamath, Keenan, Somers and Sorenson
    (2024), *Large Language Models: A Deep Dive*, Springer, Ch 1,
    Word2Vec; the section is not present in the 2024 PDF, so the
    objective is implemented exactly as the spec line states (Mikolov
    et al. 2013).

    Examples
    --------
    >>> import math
    >>> out = kamath_word2vec_skipgram([0], [1], [[0.0], [0.0]],
    ...                                [[0.0], [0.0]])
    >>> abs(out["estimate"] + math.log(2)) < 1e-12
    True
    >>> out2 = kamath_word2vec_skipgram([0, 1], [1, 0],
    ...     [[1.0], [0.0]], [[1.0], [0.0]])
    >>> want = -math.log(math.exp(1.0) + 1.0) - math.log(2.0)
    >>> abs(out2["log_likelihood"] - want) < 1e-12
    True
    """
    c = np.atleast_1d(np.asarray(center_indices)).ravel().astype(int)
    o = np.atleast_1d(np.asarray(context_indices)).ravel().astype(int)
    V = np.atleast_2d(np.asarray(V, dtype=float))
    U = np.atleast_2d(np.asarray(U, dtype=float))
    if c.size != o.size:
        raise ValueError(
            f"{c.size} centers against {o.size} context words; the "
            "objective sums over PAIRS.")
    if c.size == 0:
        raise ValueError("no (center, context) pairs supplied.")
    if V.shape != U.shape:
        raise ValueError(
            f"the input embeddings are {V.shape} and the output "
            f"embeddings {U.shape}; skip-gram keeps one of each per "
            "vocabulary entry.")
    n_vocab = V.shape[0]
    if np.any((c < 0) | (c >= n_vocab)) or np.any((o < 0) | (o >= n_vocab)):
        raise ValueError(
            f"an index lies outside the vocabulary [0, {n_vocab - 1}].")
    if np.any(c == o):
        raise ValueError(
            "a word is listed as its own context (j = 0 is excluded "
            "from the window).")
    scores = V[c] @ U.T                      # (pairs, vocab)
    m = scores.max(axis=1, keepdims=True)
    logZ = m.ravel() + np.log(np.exp(scores - m).sum(axis=1))
    per = scores[np.arange(c.size), o] - logZ
    total = float(per.sum())
    return RichResult(payload={
        "log_likelihood": total,
        "mean_log_likelihood": float(per.mean()),
        "per_pair": [float(v) for v in per],
        "probabilities": [float(np.exp(v)) for v in per],
        "estimate": float(per.mean()),
        "vocab_size": int(n_vocab),
        "n": int(c.size),
        "method": "Skip-gram log-likelihood with the full softmax"})


def cheatsheet():
    return "kmw2v: sum log softmax(u_o . v_c) over pairs; full normaliser"
