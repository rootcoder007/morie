# morie.fn -- slice s03 (rootcoder007/morie)
"""Term frequency-inverse document frequency.

Source consulted: Salton, G. and Buckley, C. (1988).  Term-weighting
approaches in automatic text retrieval.  *Information Processing and
Management* 24(5), 513-523, and Sparck Jones, K. (1972).  A statistical
interpretation of term specificity and its application in retrieval.
*Journal of Documentation* 28(1), 11-21, which introduced the idf
factor.  The weight is

    w_(t,d) = tf(t, d) * log( N / df(t) )

with tf the count of t in d, df the number of documents containing t and
N the collection size.  Neither is open access; the weight is quoted in
its standard published form.

A term appearing in every document gets idf = log(1) = 0 and so weight
exactly zero -- which is the intended behaviour, not a degenerate case,
and is why the smoothed variant log(1 + N/df) is offered separately
rather than substituted silently.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["tfidf"]


def tfidf(docs, smooth=False, sublinear=False):
    """TF-IDF matrix over a list of tokenised documents.

    Returns
    -------
    estimate : the weight of the first term in the first document
    W        : the weight matrix, documents in rows
    vocab    : the sorted term list
    idf, df  : the inverse document frequency and document frequency
    """
    D = [[str(t) for t in d] for d in docs]
    N = len(D)
    vocab = []
    for d in D:
        for t in d:
            if t not in vocab:
                vocab.append(t)
    vocab = sorted(vocab)
    V = len(vocab)
    df = [0.0] * V
    for j in range(V):
        for d in D:
            if vocab[j] in d:
                df[j] += 1.0
    idf = []
    for j in range(V):
        if smooth:
            idf.append(math.log(1.0 + N / df[j]) if df[j] > 0.0 else 0.0)
        else:
            idf.append(math.log(N / df[j]) if df[j] > 0.0 else 0.0)
    W = [[0.0] * V for _ in range(N)]
    for i in range(N):
        for j in range(V):
            tf = 0.0
            for t in D[i]:
                if t == vocab[j]:
                    tf += 1.0
            if sublinear and tf > 0.0:
                tf = 1.0 + math.log(tf)
            W[i][j] = tf * idf[j]
    return RichResult(
        title="TF-IDF",
        summary_lines=[("documents", N), ("terms", V)],
        payload={
            "estimate": W[0][0] if N and V else float("nan"),
            "W": W,
            "vocab": vocab,
            "idf": idf,
            "df": df,
            "n": N,
            "method": "TF-IDF weighting (Sparck Jones 1972; Salton and Buckley 1988)",
        },
    )


def cheatsheet():
    return "tfidfV: TF-IDF"
