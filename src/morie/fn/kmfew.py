# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Few-shot exemplar selection by similarity to the query."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_few_shot_exemplar_selection"]


def kamath_few_shot_exemplar_selection(D, query_embed, K, metric="cosine"):
    """D_K = TopK_{d in D} sim(embed(x), embed(d)).

    ``D`` is the pool of exemplar embeddings, shape (n, d); the
    embeddings are assumed already computed (the book's embed() is the
    caller's tokenizer/encoder, not this function's business). Ties are
    broken by the smaller pool index so the selection is reproducible.

    Reference: the worklist cites Kamath, Keenan, Somers and Sorenson
    (2024), *Large Language Models: A Deep Dive*, Springer, Ch 3,
    few-shot exemplar selection.

    Examples
    --------
    >>> out = kamath_few_shot_exemplar_selection(
    ...     [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [1.0, 0.0], 2)
    >>> out["selected"]
    [0, 2]
    >>> abs(out["similarities"][1] - 1 / 2 ** 0.5) < 1e-12
    True
    """
    D = np.atleast_2d(np.asarray(D, dtype=float))
    q = np.atleast_1d(np.asarray(query_embed, dtype=float)).ravel()
    K = int(K)
    n = D.shape[0]
    if D.ndim != 2:
        raise ValueError("D must be a 2-D pool of exemplar embeddings.")
    if D.shape[1] != q.size:
        raise ValueError(
            f"exemplars have width {D.shape[1]} but the query has "
            f"{q.size}; they are not in the same embedding space.")
    if not 1 <= K <= n:
        raise ValueError(f"K must lie in [1, {n}]; got {K}.")
    if metric not in ("cosine", "dot"):
        raise ValueError("metric must be 'cosine' or 'dot'.")

    if metric == "cosine":
        nq = np.linalg.norm(q)
        nd = np.linalg.norm(D, axis=1)
        if nq == 0 or np.any(nd == 0):
            raise ValueError(
                "a zero embedding has no direction, so cosine "
                "similarity is undefined; use metric='dot'.")
        sims = (D @ q) / (nd * nq)
    else:
        sims = D @ q

    # -sims sorts descending; 'stable' keeps ties in pool order.
    order = np.argsort(-sims, kind="stable")[:K]
    sel = [int(i) for i in order]
    return RichResult(payload={
        "selected": sel,
        "similarities": [float(sims[i]) for i in sel],
        "all_similarities": [float(v) for v in sims],
        "estimate": float(sims[sel[0]]),
        "K": K, "metric": metric, "n": n,
        "method": "Top-K few-shot exemplar selection by similarity"})


def cheatsheet():
    return "kmfew: top-K exemplars by cosine/dot similarity to the query"
