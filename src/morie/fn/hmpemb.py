# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pretrained word embeddings as initialisation."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_pretrained_embeddings"]


def _lcg_row(dim, seed, scale):
    s = int(seed) % 2**32
    out = np.empty(dim)
    for i in range(dim):
        s = (1664525 * s + 1013904223) % 2**32
        out[i] = ((s + 0.5) / 2**32 * 2.0 - 1.0) * scale
    return out


def geron_pretrained_embeddings(vocab, pretrained, freeze=True, seed=0, oov_scale=0.05):
    """
    Use pretrained word embeddings (e.g. GloVe) as initialisation.

    Formula: E init from pretrained matrix; optionally fine-tune

    The embedding table is usually the largest layer in a text model and
    the one with the least supervision per parameter, so initialising it
    from a corpus millions of times bigger than the task's own is where
    transfer pays most.

    Two decisions are made explicitly here rather than by default.
    COVERAGE: a word missing from the pretrained vocabulary gets a small
    random row, and if coverage is low the "pretrained" table is mostly
    noise -- the number is returned so that can be checked instead of
    assumed. FREEZING: with a small dataset, fine-tuning the table
    destroys the geometry it was given (only the words that appear move,
    so the rest drift out of alignment); ``freeze=True`` protects it and
    drops the trainable parameter count to zero.

    Parameters
    ----------
    vocab : sequence
        Tokens in index order.
    pretrained : mapping
        Token -> vector; all vectors must share a width.
    freeze : bool, default True
        Whether the table is held fixed during training.
    seed : int, default 0
        Integer-LCG seed for the out-of-vocabulary rows.
    oov_scale : float, default 0.05
        Half-width of the uniform OOV initialisation.

    Returns
    -------
    result : RichResult
        Keys: embeddings, coverage, oov, oov_indices, dim, trainable,
        n_parameters, estimate, n, method.

    Examples
    --------
    >>> r = geron_pretrained_embeddings(["cat", "zzz"], {"cat": [1.0, 0.0]})
    >>> [float(v) for v in r["embeddings"][0]]
    [1.0, 0.0]
    >>> float(r["coverage"]), r["oov"], int(r["dim"])
    (0.5, ['zzz'], 2)

    The OOV row is small but not zero, so the word is distinguishable
    from padding:

    >>> bool(0 < np.abs(r["embeddings"][1]).max() <= 0.05)
    True

    Frozen, the table trains nothing; unfrozen, all of it:

    >>> int(r["trainable"]), int(r["n_parameters"])
    (0, 4)
    >>> int(geron_pretrained_embeddings(["cat"], {"cat": [1.0, 0.0]}, freeze=False)["trainable"])
    2

    References
    ----------
    Geron Ch 14
    """
    words = list(vocab)
    if not words:
        raise ValueError("geron_pretrained_embeddings: vocab is empty")
    if not hasattr(pretrained, "get"):
        raise ValueError("geron_pretrained_embeddings: pretrained must be a mapping of token -> vector")
    if len(pretrained) == 0:
        raise ValueError("geron_pretrained_embeddings: pretrained is empty; there is nothing to transfer")
    dims = {int(np.asarray(v, dtype=float).ravel().size) for v in pretrained.values()}
    if len(dims) != 1:
        raise ValueError(f"geron_pretrained_embeddings: pretrained vectors have mixed widths {sorted(dims)}")
    dim = dims.pop()
    if dim < 1:
        raise ValueError("geron_pretrained_embeddings: pretrained vectors are empty")
    sc = float(oov_scale)
    if sc <= 0:
        raise ValueError(f"geron_pretrained_embeddings: oov_scale must be positive, got {oov_scale!r}")

    E = np.empty((len(words), dim))
    oov, oov_idx = [], []
    for i, wtok in enumerate(words):
        vec = pretrained.get(wtok)
        if vec is None:
            E[i] = _lcg_row(dim, seed + 7919 * i + 1, sc)
            oov.append(wtok)
            oov_idx.append(i)
        else:
            row = np.asarray(vec, dtype=float).ravel()
            if not np.all(np.isfinite(row)):
                raise ValueError(f"geron_pretrained_embeddings: the vector for {wtok!r} is not finite")
            E[i] = row

    cover = 1.0 - len(oov) / len(words)
    total = int(E.size)
    return RichResult(
        title="Pretrained embeddings",
        summary_lines=[("Vocabulary", len(words)), ("Dimension", int(dim)), ("Coverage", cover)],
        warnings=(["coverage is below 50 %: most rows are random, so this table is barely pretrained"] if cover < 0.5 else []),
        interpretation="Fine-tuning on a small set moves only the seen words and breaks the geometry; freeze it then.",
        payload={
            "embeddings": E,
            "coverage": cover,
            "oov": oov,
            "oov_indices": oov_idx,
            "dim": int(dim),
            "freeze": bool(freeze),
            "trainable": 0 if freeze else total,
            "n_parameters": total,
            "estimate": E,
            "n": len(words),
            "method": "Embedding table built from pretrained vectors with LCG-initialised OOV rows",
        },
    )


def cheatsheet():
    return "hmpemb: Pretrained word embeddings as initialisation"
