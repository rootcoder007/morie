# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 7.3: RAGAS answer relevance."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_ch7_answer_relevance"]


def _cosine(a, B):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(B, axis=1)
    if na == 0 or np.any(nb == 0):
        raise ValueError("a zero-length embedding has no direction, so "
                         "its cosine similarity is undefined.")
    return (B @ a) / (nb * na)


def kamath_ch7_answer_relevance(E_g, E_o, N=None):
    r"""Answer Relevance = (1/N) sum_i sim(E_{g_i}, E_o).

    ``E_g`` holds the embeddings of the N questions reverse-generated
    from the answer, ``E_o`` the embedding of the original user query;
    ``sim`` is cosine similarity, as in RAGAS. ``N`` is optional and
    only checked against the number of rows of ``E_g``.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 7, Eq 7.3, printed
    p. 300.

    Examples
    --------
    >>> out = kamath_ch7_answer_relevance([[1.0, 0.0], [0.0, 1.0]],
    ...                                   [1.0, 0.0])
    >>> out["estimate"]          # cos = 1 and 0
    0.5
    """
    G = np.atleast_2d(np.asarray(E_g, dtype=float))
    o = np.atleast_1d(np.asarray(E_o, dtype=float)).ravel()
    if G.size == 0:
        raise ValueError("no reverse-generated questions were given.")
    if G.shape[1] != o.size:
        raise ValueError(
            f"embedding widths differ: E_g is {G.shape[1]}-dimensional, "
            f"E_o is {o.size}-dimensional.")
    if N is not None and int(N) != G.shape[0]:
        raise ValueError(
            f"N = {N} contradicts the {G.shape[0]} question embeddings "
            "supplied.")
    sims = _cosine(o, G)
    return RichResult(payload={
        "estimate": float(sims.mean()),
        "similarities": [float(v) for v in sims],
        "n": int(G.shape[0]),
        "method": "RAGAS answer relevance (Kamath Eq 7.3)"})


def cheatsheet():
    return "km112: mean cosine(reverse-generated question, query)"
