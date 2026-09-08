# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Class-based TF-IDF (Grootendorst 2022; Alammar Ch 5)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["alammar_c_tfidf"]


def alammar_c_tfidf(term_counts_by_class, corpus_freq=None, A=None):
    """c-TF-IDF(t, c) = f_{t,c} * log(1 + A / f_t), with f_t the
    term's frequency ACROSS classes and A the mean class size.

    ``term_counts_by_class`` is a classes x terms count matrix. A
    defaults to the mean total count per class, per the BERTopic
    paper; f_t defaults to the column sums.

    References: Alammar and Grootendorst, Ch 5; Grootendorst (2022),
    arXiv:2203.05794, Eq 3.
    """
    M = np.atleast_2d(np.asarray(term_counts_by_class, dtype=float))
    if np.any(M < 0):
        raise ValueError("counts must be non-negative.")
    f_t = (np.asarray(corpus_freq, dtype=float) if corpus_freq is not None
           else M.sum(axis=0))
    if len(f_t) != M.shape[1]:
        raise ValueError("corpus_freq must have one entry per term.")
    if np.any(f_t <= 0):
        raise ValueError(
            "a term with zero corpus frequency cannot be weighted; drop "
            "it before calling.")
    a = float(A) if A is not None else float(M.sum(axis=1).mean())
    W = M * np.log1p(a / f_t)[None, :]
    return RichResult(payload={
        "weights": [[float(v) for v in r] for r in W],
        "A": a, "corpus_freq": [float(v) for v in f_t],
        "top_term_per_class": [int(i) for i in np.argmax(W, axis=1)],
        "estimate": float(W[0, 0]), "n": M.shape[0],
        "method": "c-TF-IDF (Grootendorst 2022, Eq 3)"})


def cheatsheet():
    return "alctf: class counts times log(1 + A/f_t), BERTopic Eq 3"


# compact alias per ledger/NAMING.md
alammarctfidf = alammar_c_tfidf
