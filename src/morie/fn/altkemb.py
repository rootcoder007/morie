# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Token embedding lookup (Alammar Ch 2)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["alammar_token_embedding_lookup"]


def alammar_token_embedding_lookup(ids, E_tok):
    """E_tok[ids]: rows of the V x d table, in sequence order.

    An out-of-vocabulary id is refused, not clamped -- clamping is how
    silent garbage enters a pipeline.

    References: Alammar and Grootendorst, Ch 2.

    Examples
    --------
    >>> alammar_token_embedding_lookup([1, 0],
    ...     [[1.0, 2.0], [3.0, 4.0]])["embeddings"]
    [[3.0, 4.0], [1.0, 2.0]]
    """
    E = np.atleast_2d(np.asarray(E_tok, dtype=float))
    ids = np.atleast_1d(np.asarray(ids)).astype(int)
    if np.any((ids < 0) | (ids >= E.shape[0])):
        bad = ids[(ids < 0) | (ids >= E.shape[0])][0]
        raise ValueError(
            f"token id {bad} is outside the vocabulary of {E.shape[0]}.")
    out = E[ids]
    return RichResult(payload={
        "embeddings": [[float(v) for v in r] for r in out],
        "estimate": float(out[0, 0]), "vocab_size": E.shape[0],
        "dim": E.shape[1], "n": len(ids),
        "method": "Token embedding lookup (Alammar Ch 2)"})


def cheatsheet():
    return "altkemb: E[ids] row lookup with OOV refused"
