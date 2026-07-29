# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Burkov Ch 2: term frequency."""

import numpy as np

from ._richresult import RichResult

__all__ = ["burkov_term_frequency"]


def burkov_term_frequency(term, document, normalise=False):
    """TF(t, d) = count of t in d, optionally divided by |d|.

    References: Burkov LM (2025), Ch 2, term frequency.

    Examples
    --------
    >>> burkov_term_frequency("a", ["a", "b", "a"])["estimate"]
    2.0
    >>> burkov_term_frequency("a", ["a", "b", "a"], normalise=True)["estimate"]
    0.6666666666666666
    """
    doc = [str(t) for t in np.atleast_1d(np.asarray(document, dtype=object))]
    if not doc:
        raise ValueError("the document is empty.")
    c = doc.count(str(term))
    est = c / len(doc) if normalise else float(c)
    return RichResult(payload={
        "estimate": float(est), "count": c, "doc_length": len(doc),
        "normalised": bool(normalise), "n": len(doc),
        "method": "Term frequency (Burkov Ch 2)"})


def cheatsheet():
    return "bktf: term frequency count(t in d) (Burkov Ch 2)"
