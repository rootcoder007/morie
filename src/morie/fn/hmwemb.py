# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Word embeddings: dense vector representations learned per token."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_word_embeddings"]


def geron_word_embeddings(vocab, d=8, E=None, seed=0):
    """
    Word embeddings: dense vector representations learned per token.

    Formula: E in R^(V x d); token t -> E[t]

    An embedding layer is a lookup table, not a matrix multiply: row
    ``E[i]`` *is* the representation of token i, and the "multiplication"
    by a one-hot vector is only how it is written down. This builds the
    table (deterministic LCG initialisation, or a supplied pretrained
    matrix), returns the lookup, and computes the cosine-similarity
    matrix that makes the geometry inspectable.

    Duplicate vocabulary entries are an error: two rows for one token
    means the model learns two unrelated vectors for the same word and
    the lookup becomes ambiguous.

    Parameters
    ----------
    vocab : sequence
        Distinct tokens (non-empty).
    d : int, default 8
        Embedding width (>= 1); ignored when `E` is supplied.
    E : array-like, optional
        Pretrained embedding matrix of shape (V, d).
    seed : int, default 0
        LCG seed for the initialisation.

    Returns
    -------
    result : RichResult
        Keys: E, lookup, index, norms, similarity, n_params,
        estimate, n, method.

    Examples
    --------
    >>> r = geron_word_embeddings(["cat", "dog", "the"], d=4)
    >>> r["E"].shape
    (3, 4)
    >>> int(r["n_params"])
    12
    >>> bool((r["lookup"]("dog") == r["E"][1]).all())
    True
    >>> r["index"]["the"]
    2

    Similarities are cosines, so the diagonal is exactly 1:

    >>> [round(float(r["similarity"][i, i]), 12) for i in range(3)]
    [1.0, 1.0, 1.0]

    A pretrained table is used as given -- here two tokens share a
    direction and are therefore perfectly similar:

    >>> r2 = geron_word_embeddings(["a", "b"], E=[[1.0, 0.0], [2.0, 0.0]])
    >>> round(float(r2["similarity"][0, 1]), 12)
    1.0

    References
    ----------
    Géron Ch 14
    """
    toks = list(vocab)
    if not toks:
        raise ValueError("geron_word_embeddings: vocab is empty")
    if len(set(toks)) != len(toks):
        dup = sorted({t for t in toks if toks.count(t) > 1}, key=str)
        raise ValueError(f"geron_word_embeddings: duplicate vocabulary entries {dup}; each token needs exactly one row")
    V = len(toks)

    if E is None:
        k = int(d)
        if k < 1:
            raise ValueError(f"geron_word_embeddings: d must be >= 1, got {k}")
        s = int(seed) % 2**32
        flat = np.empty(V * k)
        scale = 1.0 / np.sqrt(k)
        for i in range(V * k):
            s = (1664525 * s + 1013904223) % 2**32
            flat[i] = (2.0 * ((s + 0.5) / 2**32) - 1.0) * scale
        M = flat.reshape(V, k)
    else:
        M = np.asarray(E, dtype=float)
        if M.ndim != 2:
            raise ValueError("geron_word_embeddings: E must be a 2-D (V, d) matrix")
        if M.shape[0] != V:
            raise ValueError(f"geron_word_embeddings: vocab has {V} tokens but E has {M.shape[0]} rows")
        if not np.all(np.isfinite(M)):
            raise ValueError("geron_word_embeddings: E contains non-finite values")
        k = M.shape[1]

    index = {t: i for i, t in enumerate(toks)}

    def lookup(token, _M=M, _idx=index):
        if isinstance(token, (list, tuple)):
            missing = [t for t in token if t not in _idx]
            if missing:
                raise ValueError(f"lookup: token(s) {missing} are not in the vocabulary")
            return _M[[_idx[t] for t in token]]
        if token not in _idx:
            raise ValueError(f"lookup: token {token!r} is not in the vocabulary")
        return _M[_idx[token]]

    norms = np.sqrt(np.sum(M * M, axis=1))
    safe = np.where(norms > 0, norms, 1.0)
    U = M / safe[:, None]
    sim = U @ U.T

    return RichResult(
        title="Word embedding table",
        summary_lines=[("Vocabulary", V), ("Dimension", int(k)), ("Parameters", int(V * k))],
        interpretation=(
            "Embeddings put tokens in a space where distance means something; the table is pure "
            "parameters (V*d of them), which is why vocabulary size dominates a small model's footprint."
        ),
        payload={
            "E": M,
            "lookup": lookup,
            "index": index,
            "vocab": toks,
            "norms": norms,
            "similarity": sim,
            "n_params": int(V * k),
            "d": int(k),
            "estimate": float(V * k),
            "n": int(V),
            "method": "Embedding lookup table with cosine-similarity geometry",
        },
    )


def cheatsheet():
    return "hmwemb: Word embeddings: dense vector representations learned per token"
