# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""BERTopic pipeline: embed, reduce, cluster, c-TF-IDF
(Grootendorst 2022; Alammar Ch 5)."""

import numpy as np

from ._richresult import RichResult
from .alctf import alammar_c_tfidf
from .alhds import alammar_hdbscan_cluster

__all__ = ["alammar_bertopic_pipeline"]


def alammar_bertopic_pipeline(documents, embeddings, min_cluster_size=2):
    """topics = c-TF-IDF over clusters of (reduced) document
    embeddings. Embeddings are supplied (the encoder is a model the
    caller owns); the reduction, clustering and topic-word scoring are
    computed natively here, and every step's output is returned so
    the pipeline can be audited joint by joint.

    References: Alammar and Grootendorst, Ch 5; Grootendorst (2022).
    """
    docs = [[str(w) for w in d] for d in documents]
    E = np.atleast_2d(np.asarray(embeddings, dtype=float))
    if E.shape[0] != len(docs):
        raise ValueError("need one embedding per document.")
    if len(docs) < 4:
        raise ValueError("need at least 4 documents to cluster.")
    # linear reduction to 2 principal axes (deterministic SVD)
    C = E - E.mean(axis=0)
    U, S, Vt = np.linalg.svd(C, full_matrices=False)
    Z = C @ Vt[:2].T
    cl = alammar_hdbscan_cluster(Z, min_cluster_size=min_cluster_size,
                                 min_samples=1)
    labels = cl["labels"]
    vocab = sorted({w for d in docs for w in d})
    widx = {w: i for i, w in enumerate(vocab)}
    clusters = sorted({l for l in labels if l >= 0})
    if not clusters:
        raise ValueError(
            "every document came out as noise; loosen min_cluster_size.")
    M = np.zeros((len(clusters), len(vocab)))
    for lab, d in zip(labels, docs):
        if lab >= 0:
            row = clusters.index(lab)
            for w in d:
                M[row, widx[w]] += 1
    ct = alammar_c_tfidf(M)
    W = np.asarray(ct["weights"])
    top_words = {int(clusters[i]): vocab[int(np.argmax(W[i]))]
                 for i in range(len(clusters))}
    return RichResult(payload={
        "labels": labels, "reduced": [[float(v) for v in r] for r in Z],
        "topic_top_word": top_words,
        "n_topics": len(clusters), "vocabulary": vocab,
        "estimate": float(len(clusters)), "n": len(docs),
        "method": "BERTopic: reduce, cluster, c-TF-IDF (Grootendorst 2022)"})


def cheatsheet():
    return "albtm: SVD reduce, density cluster, c-TF-IDF topic words"
