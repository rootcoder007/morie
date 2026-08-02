# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bag-of-words vector (Alammar Ch 1)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["alammar_bag_of_words"]


def alammar_bag_of_words(tokens, vocab):
    """bow[v] = count of vocabulary word v in the document.

    Out-of-vocabulary tokens are counted and reported, not silently
    dropped into nothing -- their number is exactly what the vector
    fails to represent.

    Examples
    --------
    >>> out = alammar_bag_of_words(["a", "b", "a", "z"], ["a", "b", "c"])
    >>> out["bow_vector"]
    [2, 1, 0]
    >>> out["oov_count"]
    1
    """
    toks = [str(t) for t in np.atleast_1d(np.asarray(tokens, dtype=object))]
    voc = [str(v) for v in np.atleast_1d(np.asarray(vocab, dtype=object))]
    if len(set(voc)) != len(voc):
        raise ValueError("the vocabulary contains duplicates.")
    if not voc:
        raise ValueError("the vocabulary is empty.")
    idx = {v: i for i, v in enumerate(voc)}
    bow = [0] * len(voc)
    oov = 0
    for t in toks:
        if t in idx:
            bow[idx[t]] += 1
        else:
            oov += 1
    return RichResult(payload={
        "bow_vector": bow, "oov_count": oov,
        "estimate": float(bow[0]), "vocab_size": len(voc),
        "n": len(toks),
        "method": "Bag-of-words counts over a fixed vocabulary (Alammar Ch 1)"})


def cheatsheet():
    return "albow: per-vocab counts, OOV counted not dropped"
