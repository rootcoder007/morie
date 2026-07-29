# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Ch 7: dense passage retrieval (DPR) top-k."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_dense_passage_retrieval"]


def kamath_dense_passage_retrieval(q_embed, p_embeds, k):
    r"""score(q, p) = E_Q(q)^T E_P(p); top-k by descending score.

    A bi-encoder scores with a plain DOT product, not a cosine -- the
    passage norm carries information the DPR objective trains -- so no
    normalization is applied here. Ties keep index order (stable
    sort).

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 7, Dense Retrieval;
    Karpukhin et al. (2020).

    Examples
    --------
    >>> out = kamath_dense_passage_retrieval(
    ...     [1.0, 0.0], [[1.0, 0.0], [0.0, 1.0], [2.0, 0.0]], 2)
    >>> out["top_k_indices"], out["top_k_scores"]
    ([2, 0], [2.0, 1.0])
    """
    q = np.atleast_1d(np.asarray(q_embed, dtype=float)).ravel()
    P = np.atleast_2d(np.asarray(p_embeds, dtype=float))
    if P.size == 0:
        raise ValueError("the passage index is empty.")
    if P.shape[1] != q.size:
        raise ValueError(
            f"the query is {q.size}-dimensional but the passages are "
            f"{P.shape[1]}-dimensional.")
    kk = int(k)
    if not (1 <= kk <= P.shape[0]):
        raise ValueError(
            f"k = {kk} must lie in [1, {P.shape[0]}] for this index.")
    s = P @ q
    order = np.argsort(-s, kind="stable")[:kk]
    return RichResult(payload={
        "estimate": float(s[order[0]]),
        "top_k_indices": [int(i) for i in order],
        "top_k_scores": [float(s[i]) for i in order],
        "scores": [float(v) for v in s], "k": kk, "n": int(P.shape[0]),
        "method": "dense passage retrieval top-k (Kamath Ch 7)"})


def cheatsheet():
    return "kmdpr: bi-encoder dot products, stable top-k"
