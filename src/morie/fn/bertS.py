# morie.fn -- function file (rootcoder007/morie)
"""BERTScore precision, recall and F1."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["bertscore"]


def bertscore(reference, candidate, idf=None):
    """Greedy cosine matching between contextual token embeddings.

    Rather than counting exact n-gram overlaps, BERTScore represents each
    token by a contextual embedding and matches greedily on cosine
    similarity, each token pairing with its most similar token in the
    other sentence.  For a reference x of length k and a candidate xhat
    of length l with pre-normalised embeddings, so that cosine similarity
    is the inner product,

        R = (1/|x|)    sum_{x_i in x}       max_{xhat_j} x_i' xhat_j
        P = (1/|xhat|) sum_{xhat_j in xhat} max_{x_i}    x_i' xhat_j
        F = 2 P R / (P + R)

    Optional inverse-document-frequency weights re-weight the recall sum,
    R = sum_i idf(x_i) max_j x_i' xhat_j / sum_i idf(x_i), so that rare
    tokens count for more.

    Parameters
    ----------
    reference : array-like, shape (k, d)
        Reference token embeddings; normalised internally.
    candidate : array-like, shape (l, d)
        Candidate token embeddings; normalised internally.
    idf : array-like or None
        Importance weight per reference token, length k.

    Returns
    -------
    RichResult
        ``P``, ``R``, ``F``, ``recallmatch``, ``precmatch``, ``k``, ``l``,
        ``d``.

    References
    ----------
    Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q. and Artzi, Y.
    (2020), "BERTScore: evaluating text generation with BERT",
    International Conference on Learning Representations;
    arXiv:1904.09675.  Sect. 3 gives the cosine similarity of
    pre-normalised embeddings as the inner product and the three
    equations for R_BERT, P_BERT and F_BERT reproduced above, together
    with the idf-weighted recall.  Read from the ar5iv rendering of the
    arXiv source.
    """
    import math
    X = C.mat(reference)
    Y = C.mat(candidate)
    k, d = len(X), len(X[0])
    l = len(Y)
    if len(Y[0]) != d:
        raise ValueError("embeddings must share their dimension")

    def unit(v):
        nv = math.sqrt(sum(t * t for t in v))
        if nv == 0.0:
            raise ValueError("embeddings must be non-zero")
        return [t / nv for t in v]

    Xn = [unit(r) for r in X]
    Yn = [unit(r) for r in Y]
    Sim = [[sum(Xn[i][t] * Yn[j][t] for t in range(d)) for j in range(l)]
           for i in range(k)]
    rm = [max(Sim[i]) for i in range(k)]
    pm = [max(Sim[i][j] for i in range(k)) for j in range(l)]
    if idf is None:
        w = [1.0] * k
    else:
        w = C.vec(idf)
        if len(w) != k:
            raise ValueError("idf must have one weight per reference token")
        if sum(w) <= 0.0:
            raise ValueError("idf weights must not all be zero")
    R = sum(w[i] * rm[i] for i in range(k)) / sum(w)
    P = sum(pm) / l
    Fv = 0.0 if P + R == 0.0 else 2.0 * P * R / (P + R)
    return RichResult(payload={
        "P": P, "R": R, "F": Fv, "recallmatch": rm, "precmatch": pm,
        "k": k, "l": l, "d": d,
        "method": "BERTScore greedy cosine matching (Zhang et al. 2020)"})


def cheatsheet():
    return "bertS: BERTScore precision, recall and F1."
