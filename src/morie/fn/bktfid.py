# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Burkov Ch 2: TF-IDF."""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["burkov_tf_idf"]


def burkov_tf_idf(term, document, corpus):
    """TF-IDF = TF(t, d) * log(|D| / df(t)).

    A term appearing in NO document has df = 0 and an undefined IDF;
    that is refused, since it can only happen when the query document
    is not in the corpus, which is a caller error worth hearing about.

    References: Burkov LM (2025), Ch 2, TF-IDF.

    Examples
    --------
    >>> corpus = [["a", "b"], ["b", "c"], ["b"]]
    >>> round(burkov_tf_idf("a", ["a", "b"], corpus)["estimate"], 10)
    1.0986122887
    """
    t = str(term)
    doc = [str(x) for x in np.atleast_1d(np.asarray(document, dtype=object))]
    if not corpus:
        raise ValueError("the corpus is empty.")
    docs = [[str(x) for x in np.atleast_1d(np.asarray(d, dtype=object))]
            for d in corpus]
    tf = doc.count(t)
    df = sum(1 for d in docs if t in d)
    if df == 0:
        raise ValueError(
            f"term {t!r} appears in no corpus document, so IDF is "
            "undefined; is the query document part of the corpus?")
    idf = math.log(len(docs) / df)
    return RichResult(payload={
        "estimate": tf * idf, "tf": tf, "df": df, "idf": idf,
        "n_documents": len(docs), "n": len(doc),
        "method": "TF-IDF (Burkov Ch 2)"})


def cheatsheet():
    return "bktfid: TF-IDF tf * log(|D|/df) (Burkov Ch 2)"
