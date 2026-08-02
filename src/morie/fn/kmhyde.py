# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""HyDE: retrieve with an LLM-generated hypothetical answer as the
query."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_hyde_hypothetical_doc"]


def kamath_hyde_hypothetical_doc(query, model, embeddings, embed=None, k=3):
    """y_hypo = LLM(x); retrieve top-k by sim(embed(y_hypo), embed(d)).

    Orchestration only -- the two learned pieces are the caller's:
    ``model(query) -> str`` writes the hypothetical answer and
    ``embed(text) -> vector`` encodes it. ``embeddings`` is the corpus
    side, either a matrix of document vectors or a dict
    ``{doc_id: vector}``. If ``embed`` is omitted the model must
    return a vector directly, and that is checked rather than assumed.

    Reference: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 7, HyDE.

    Examples
    --------
    >>> docs = [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]]
    >>> out = kamath_hyde_hypothetical_doc(
    ...     "what colour", lambda q: "red",
    ...     docs, embed=lambda t: [1.0, 0.0], k=2)
    >>> out["retrieved"]
    [0, 2]
    >>> out["hypothetical"]
    'red'
    """
    if not callable(model):
        raise ValueError("model must be callable query -> hypothetical text.")
    if embed is not None and not callable(embed):
        raise ValueError("embed must be callable text -> vector.")

    if isinstance(embeddings, dict):
        if not embeddings:
            raise ValueError("the document embedding table is empty.")
        ids = list(embeddings.keys())
        D = np.atleast_2d(np.asarray([embeddings[i] for i in ids],
                                     dtype=float))
    else:
        D = np.atleast_2d(np.asarray(embeddings, dtype=float))
        ids = list(range(D.shape[0]))
        if D.shape[0] == 0:
            raise ValueError("the document embedding matrix is empty.")

    hypo = model(query)
    if hypo is None:
        raise ValueError("the model returned no hypothetical document.")
    if embed is not None:
        q = embed(hypo)
    else:
        q = hypo
    q = np.atleast_1d(np.asarray(q, dtype=float)).ravel()
    if q.size != D.shape[1]:
        raise ValueError(
            f"the hypothetical embedding has width {q.size} but the "
            f"corpus has {D.shape[1]}; pass embed= if the model "
            "returns text.")
    k = int(k)
    if not 1 <= k <= D.shape[0]:
        raise ValueError(f"k must lie in [1, {D.shape[0]}].")

    nq = np.linalg.norm(q)
    nd = np.linalg.norm(D, axis=1)
    if nq == 0 or np.any(nd == 0):
        raise ValueError(
            "a zero embedding has no direction; cosine similarity is "
            "undefined.")
    sims = (D @ q) / (nd * nq)
    order = np.argsort(-sims, kind="stable")[:k]
    return RichResult(payload={
        "retrieved": [ids[i] for i in order],
        "similarities": [float(sims[i]) for i in order],
        "hypothetical": hypo,
        "estimate": float(sims[order[0]]),
        "k": k, "n": int(D.shape[0]),
        "method": "HyDE retrieval via a hypothetical document"})


def cheatsheet():
    return "kmhyde: embed LLM(x)'s fake answer, retrieve top-k by cosine"
