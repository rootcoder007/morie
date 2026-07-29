# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Embedding lookup."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_embedding_lookup"]

_METHOD = "Embedding table lookup"


def geron_embedding_lookup(ids, E):
    r"""Map token ids to dense vectors.

    .. math::
        e = E[\text{id}], \qquad E \in \mathbb R^{V \times d}

    An embedding layer is a lookup, not a matrix product -- it is what
    a one-hot vector times ``E`` would give, computed by indexing
    instead.  For a 50 000-word vocabulary that is the difference
    between reading 300 numbers and multiplying by a 50 000 x 300
    matrix, which is why no framework ever materialises the one-hot.

    Repeated ids share the *same* row, so a gradient arriving at two
    positions of the same token accumulates into one place; that shared
    identity is what lets rare words learn from every occurrence.

    Parameters
    ----------
    ids : array-like of int
        Token ids in ``0 .. V-1``. Scalar, 1-D or 2-D (batch, time).
    E : array-like, shape (V, d)
        Embedding table.

    Returns
    -------
    RichResult
        Payload keys ``embeddings``, ``vocab_size``, ``dim``,
        ``n_unique``, ``n_parameters``, ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 14, Word Embeddings section.

    Examples
    --------
    Rows 0 and 2 of a three-word table:

    >>> E = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    >>> r = geron_embedding_lookup([0, 2], E)
    >>> r["embeddings"]
    [[1.0, 0.0], [1.0, 1.0]]
    >>> (r["vocab_size"], r["dim"], r["n_parameters"])
    (3, 2, 6)

    A repeated id yields the identical vector -- one row, two uses:

    >>> r2 = geron_embedding_lookup([1, 1], E)
    >>> r2["embeddings"][0] == r2["embeddings"][1]
    True
    >>> r2["n_unique"]
    1

    An out-of-range id is an error, not a silently wrapped negative
    index:

    >>> geron_embedding_lookup([-1], E)
    Traceback (most recent call last):
        ...
    ValueError: token ids must lie in [0, 2], got range [-1, -1].
    """
    T = np.atleast_2d(np.asarray(E, dtype=float))
    if T.ndim != 2 or T.size == 0:
        raise ValueError(f"E must be a non-empty 2-D table of shape (V, d), got {T.shape}.")
    if not np.all(np.isfinite(T)):
        raise ValueError("E must be finite.")
    V, d = T.shape
    I = np.asarray(ids)
    if I.size == 0:
        raise ValueError("ids is empty.")
    if not np.issubdtype(I.dtype, np.integer):
        if not np.all(np.asarray(I, dtype=float) == np.floor(np.asarray(I, dtype=float))):
            raise ValueError("token ids must be integers.")
        I = I.astype(int)
    if I.min() < 0 or I.max() >= V:
        raise ValueError(f"token ids must lie in [0, {V - 1}], got range [{I.min()}, {I.max()}].")

    out = T[I]

    return RichResult(
        title="Embedding lookup",
        summary_lines=[("Vocab", int(V)), ("Dim", int(d)), ("Ids", int(I.size))],
        payload={
            "embeddings": out.tolist(),
            "vocab_size": int(V),
            "dim": int(d),
            "n_unique": int(np.unique(I).size),
            "n_parameters": int(V * d),
            "ids": I.tolist(),
            "estimate": out.tolist(),
            "n": int(I.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "gremb: e = E[id]; a lookup, not a one-hot matrix product; repeats share the row"
