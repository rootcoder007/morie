# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""XLNet: permutation-based autoregressive pretraining."""

from . import _array_core as np

from ._richresult import RichResult
from .hmsftm import geron_softmax_function

__all__ = ["geron_xlnet", "permutation_masks"]


def permutation_masks(perm):
    """Two-stream attention masks for a factorisation order.

    ``content[t, j] = 1`` when position j precedes-or-equals t in `perm`
    (the content stream may see the token itself), while
    ``query[t, j] = 1`` only for strictly earlier positions -- the query
    stream must not see the token it is predicting. That strict/non-strict
    pair is the whole two-stream trick.

    >>> c, q = permutation_masks([1, 0, 2])
    >>> c[1].tolist()
    [0.0, 1.0, 0.0]
    >>> q[1].tolist()
    [0.0, 0.0, 0.0]
    """
    p = np.asarray(perm).ravel().astype(int)
    T = p.size
    rank = np.empty(T, dtype=int)
    rank[p] = np.arange(T)
    content = np.zeros((T, T))
    query = np.zeros((T, T))
    for t in range(T):
        for j in range(T):
            if rank[j] <= rank[t]:
                content[t, j] = 1.0
            if rank[j] < rank[t]:
                query[t, j] = 1.0
    return content, query


def geron_xlnet(X, n_layers=1, vocab_size=None, d_model=8, seed=0):
    """
    XLNet: permutation-based autoregressive pretraining.

    Formula: maximize E_pi sum_t log P(x_{pi_t} | x_{pi_<t})

    XLNet keeps an autoregressive factorisation -- so, unlike BERT, it
    never assumes the masked tokens are conditionally independent -- but
    factorises in a *permuted* order, which lets every token condition on
    context from both sides. Implemented here as the mechanism that makes
    that possible:

    * a factorisation order is drawn from a deterministic LCG;
    * :func:`permutation_masks` builds the **two-stream** masks: the
      content stream sees ``x_{pi<=t}``, the query stream only
      ``x_{pi<t}``. Without that split the model could read the answer
      off the token it is predicting;
    * each conditional ``P(x_{pi_t} | x_{pi_<t})`` is scored from the
      visible context under an embedding-based softmax head
      (delegated to :func:`morie.fn.hmsftm.geron_softmax_function`), and
      the per-token log-probabilities are summed.

    The sum over the permutation covers every position exactly once,
    whatever the order -- that invariant is checked.

    Parameters
    ----------
    X : array-like of int
        Token ids of one sequence (length >= 2).
    n_layers : int, default 1
        Encoder depth (>= 1); reported and used to scale the context mix.
    vocab_size : int, optional
        Vocabulary; defaults to ``max(X) + 1``.
    d_model : int, default 8
        Embedding width (>= 1).
    seed : int, default 0
        LCG seed for the permutation and the embeddings.

    Returns
    -------
    result : RichResult
        Keys: permutation, content_mask, query_mask, logprobs,
        total_logprob, perplexity, estimate, n, method.

    Examples
    --------
    >>> import numpy as np
    >>> r = geron_xlnet([0, 1, 2, 1], n_layers=1, vocab_size=3)
    >>> sorted(int(v) for v in r["permutation"])
    [0, 1, 2, 3]
    >>> float(np.diag(r["query_mask"]).sum())
    0.0
    >>> float(np.diag(r["content_mask"]).sum())
    4.0
    >>> bool(r["total_logprob"] < 0)
    True

    The first token in the factorisation order has no context at all, so
    its query-stream row is empty:

    >>> first = int(r["permutation"][0])
    >>> float(r["query_mask"][first].sum())
    0.0

    References
    ----------
    Géron Ch 15
    """
    x = np.asarray(X).ravel()
    if x.size < 2:
        raise ValueError("geron_xlnet: need at least 2 tokens to have a conditional to predict")
    if not np.all(np.equal(np.mod(x.astype(float), 1), 0)):
        raise ValueError("geron_xlnet: X must contain integer token ids")
    x = x.astype(int)
    if np.any(x < 0):
        raise ValueError("geron_xlnet: token ids must be non-negative")
    V = int(x.max()) + 1 if vocab_size is None else int(vocab_size)
    if V <= int(x.max()):
        raise ValueError(f"geron_xlnet: vocab_size {V} does not cover token id {int(x.max())}")
    d = int(d_model)
    if d < 1:
        raise ValueError(f"geron_xlnet: d_model must be >= 1, got {d}")
    L = int(n_layers)
    if L < 1:
        raise ValueError(f"geron_xlnet: n_layers must be >= 1, got {L}")

    T = x.size
    s = int(seed) % 2**32

    def _u():
        nonlocal s
        s = (1664525 * s + 1013904223) % 2**32
        return (s + 0.5) / 2**32

    perm = list(range(T))
    for i in range(T - 1, 0, -1):  # Fisher-Yates on the LCG stream
        j = int(_u() * (i + 1))
        perm[i], perm[j] = perm[j], perm[i]
    perm = np.asarray(perm, dtype=int)
    content, query = permutation_masks(perm)

    E = np.empty((V, d))
    for v in range(V):
        for k in range(d):
            E[v, k] = 2.0 * _u() - 1.0
    Wout = np.empty((d, V))
    for k in range(d):
        for v in range(V):
            Wout[k, v] = 2.0 * _u() - 1.0

    logps = np.empty(T)
    for t in range(T):
        vis = query[t] > 0
        ctx = E[x[vis]].mean(axis=0) if np.any(vis) else np.zeros(d)
        logits = (ctx * L) @ Wout
        p = np.asarray(geron_softmax_function(logits)["p"], dtype=float)
        logps[t] = float(np.log(max(float(p[x[t]]), np.finfo(float).tiny)))
    total = float(np.sum(logps))

    if sorted(perm.tolist()) != list(range(T)):
        raise ValueError("geron_xlnet: internal error, the factorisation order is not a permutation")

    return RichResult(
        title="XLNet permutation language modelling",
        summary_lines=[
            ("Tokens", T),
            ("Vocabulary", V),
            ("Total log-probability", total),
            ("Perplexity", float(np.exp(-total / T))),
        ],
        interpretation=(
            "Permuting the factorisation order gives bidirectional context without BERT's independence "
            "assumption; the query stream exists purely so a token never sees itself."
        ),
        payload={
            "permutation": perm,
            "content_mask": content,
            "query_mask": query,
            "logprobs": logps,
            "total_logprob": total,
            "perplexity": float(np.exp(-total / T)),
            "embeddings": E,
            "n_layers": L,
            "estimate": total,
            "n": int(T),
            "method": "Permutation LM: two-stream masks over a sampled factorisation order with softmax conditionals",
        },
    )


def cheatsheet():
    return "hmxln: XLNet: permutation-based autoregressive pretraining"


# compact alias per ledger/NAMING.md
geronxlnet = geron_xlnet
