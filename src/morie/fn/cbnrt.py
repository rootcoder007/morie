# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Treatment effects with text-borne confounding.

The placeholder this replaces was specified as CausalBERT: transformer
embeddings feeding a propensity and outcome model. The transformer is
not implementable natively -- it needs pretrained weights this package
does not have and would have to invent -- so the *representation* here
is a native TF-IDF and truncated singular value decomposition, and the
*causal* machinery on top is the real thing.

That substitution is stated rather than hidden, and it is not
cosmetic: a bag of words cannot represent negation, word order or
sense, so any confounding carried by those is left in. The estimand
and the estimator are unchanged; the adjustment set is weaker.

Veitch V, Sridhar D, Blei DM (2020), *Adapting text embeddings for
causal inference*, UAI 2020, arXiv:1905.12741; Roberts ME, Stewart BM,
Nielsen RA (2020), *American Journal of Political Science*
64(4):887-903, on matching with text.
"""

import math
import re

from . import _array_core as np

from ._did import add_intercept, logit_fit, logit_predict
from ._richresult import RichResult

__all__ = ["causalbert_text", "tfidf_matrix", "text_embedding"]

_METHOD = "Treatment-effect estimation with a text-derived adjustment set"
_TOKEN = re.compile(r"[a-z0-9']+")


def tfidf_matrix(texts, min_df=1, max_df=1.0, vocabulary=None):
    r"""Term frequency-inverse document frequency, computed natively.

    Uses the smoothed inverse document frequency
    :math:`\log\frac{1 + n}{1 + \mathrm{df}(w)} + 1` and L2-normalises
    each row, so documents of different lengths are comparable.

    Returns
    -------
    dict with ``matrix``, ``vocabulary``, ``document_frequency``.
    """
    docs = [str(d).lower() for d in texts]
    n = len(docs)
    if n < 1:
        raise ValueError("texts must contain at least one document.")
    toks = [_TOKEN.findall(d) for d in docs]
    if vocabulary is None:
        df = {}
        for tk in toks:
            for w in set(tk):
                df[w] = df.get(w, 0) + 1
        lo = min_df if min_df >= 1 else min_df * n
        hi = max_df * n if max_df <= 1 else max_df
        vocab = sorted(w for w, c in df.items() if lo <= c <= hi)
    else:
        vocab = list(vocabulary)
    if not vocab:
        raise ValueError(
            "no terms survived the document-frequency filter; relax min_df "
            "or max_df."
        )
    index = {w: j for j, w in enumerate(vocab)}
    tf = np.zeros((n, len(vocab)))
    for i, tk in enumerate(toks):
        for w in tk:
            j = index.get(w)
            if j is not None:
                tf[i, j] += 1.0
    dfv = (tf > 0).sum(axis=0)
    idf = np.log((1.0 + n) / (1.0 + dfv)) + 1.0
    X = tf * idf[None, :]
    nrm = np.sqrt((X ** 2).sum(axis=1, keepdims=True))
    X = X / np.where(nrm > 0, nrm, 1.0)
    return {"matrix": X, "vocabulary": vocab, "document_frequency": dfv,
            "idf": idf}


def text_embedding(texts, n_components=10, min_df=1, max_df=1.0):
    """Truncated SVD of the TF-IDF matrix (latent semantic indexing).

    Returns
    -------
    dict with ``embedding``, ``singular_values``,
    ``explained_variance_ratio``, ``vocabulary``, ``components``.
    """
    tf = tfidf_matrix(texts, min_df=min_df, max_df=max_df)
    X = tf["matrix"]
    k = int(n_components)
    if k < 1:
        raise ValueError(f"n_components must be positive; got {k}.")
    k = min(k, min(X.shape))
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    tot = float(np.sum(S ** 2))
    return {
        "embedding": U[:, :k] * S[:k],
        "singular_values": S[:k],
        "explained_variance_ratio": ((S[:k] ** 2) / tot if tot > 0
                                     else np.zeros(k)),
        "components": Vt[:k],
        "vocabulary": tf["vocabulary"],
        "n_terms": len(tf["vocabulary"]),
    }


def causalbert_text(texts, T, Y, X=None, n_components=10, trim=0.02,
                    min_df=1, max_df=1.0, alpha=0.05, embedding=None):
    r"""Average treatment effect adjusting for what the text reveals.

    The setting is one where the confounder is written down but not
    coded: clinical notes, case files, judicial reasons. The text is
    turned into a low-dimensional representation and that
    representation joins the adjustment set, after which the estimator
    is ordinary augmented inverse-probability weighting.

    **What this buys and what it does not.** Adjusting for a text
    representation identifies the effect only if the representation
    captures *all* the confounding the text carries. That assumption is
    strictly stronger than it looks and it is not testable from the
    data. Two specific ways it fails here:

    * A bag of words discards order and negation. "No history of
      psychosis" and "history of psychosis" have a cosine similarity of
      0.78 under this encoding, and **exactly 1.0** -- identical
      representations -- once ``min_df`` puts "no" below the
      document-frequency floor, which is the common default. Where the
      confounder lives in the negation, little or none of it is
      adjusted away, and the estimate moves toward the unadjusted one
      while *appearing* to have controlled for the text.
    * The number of components is a bias-variance dial with no correct
      setting. Too few and confounding is left in; too many and the
      propensity model begins to separate on document identity, at
      which point the weights explode and the estimate is driven by a
      handful of documents. ``max_weight_share`` reports how
      concentrated the weights became.

    Supplying a stronger representation through ``embedding`` is the
    intended remedy for both -- see that parameter.

    The honest framing is that this narrows the set of explanations
    rather than closing it. ``naive_difference`` is returned so the
    movement from the unadjusted contrast is visible: an adjustment
    that moves nothing has either found no confounding or failed to
    represent it, and those two look identical from the outside.

    Parameters
    ----------
    texts : sequence of str
        One document per unit. A precomputed embedding may be passed
        instead via ``embedding``.
    T : array-like
        Binary treatment.
    Y : array-like
        Outcome.
    X : array-like, optional
        Additional numeric covariates to adjust for alongside the text.
    n_components : int
        Dimension of the text representation.
    trim : float
        Propensity clipping bound.
    alpha : float
        Two-sided level.
    embedding : array-like, optional
        Use this representation instead of computing one, as an
        ``(n, k)`` array aligned with ``T`` and ``Y``.

        This is the supported route to pretrained representations, and
        it is deliberately the *only* one. Passing embeddings computed
        elsewhere -- by a transformer, a sentence encoder, a topic
        model -- keeps every causal claim in this function unchanged
        while replacing the weakest part of it. What is avoided is
        taking a dependency on a model runtime, downloading weights at
        call time, or writing them to disk: the function consumes a
        numeric matrix and nothing more, so it stays native and stays
        offline. Any representation the caller can justify is
        admissible; the identification assumption is the same one
        either way, and a better representation makes it more
        plausible rather than testable.

    Returns
    -------
    RichResult
        ``estimate`` (AIPW ATE), ``se``, ``ci_lower``/``ci_upper``,
        ``naive_difference``, ``adjustment_movement``,
        ``propensity``, ``max_weight_share``, ``n_components``,
        ``explained_variance_ratio``, ``n_trimmed``.

    References
    ----------
    Veitch V, Sridhar D, Blei DM (2020) UAI, arXiv:1905.12741.
    Roberts ME, Stewart BM, Nielsen RA (2020) *AJPS* 64(4):887-903.
    """
    t = np.asarray(T, dtype=float).ravel()
    y = np.asarray(Y, dtype=float).ravel()
    n = t.size
    if y.size != n:
        raise ValueError(f"Y has length {y.size} but T has {n}.")
    if not np.all(np.isin(t, (0.0, 1.0))):
        raise ValueError("T must be binary 0/1.")
    if not 0 <= trim < 0.5:
        raise ValueError(f"trim must lie in [0, 0.5); got {trim}.")
    if n < 10:
        raise ValueError(f"need at least 10 units; got {n}.")

    if embedding is not None:
        E = np.atleast_2d(np.asarray(embedding, dtype=float))
        if E.shape[0] != n:
            E = E.T
        if E.shape[0] != n:
            raise ValueError(
                f"embedding has {E.shape[0]} rows but there are {n} units."
            )
        emb = {"embedding": E, "explained_variance_ratio": np.full(
            E.shape[1], np.nan), "n_terms": -1, "vocabulary": [],
            "singular_values": np.full(E.shape[1], np.nan)}
    else:
        if len(texts) != n:
            raise ValueError(
                f"texts has {len(texts)} documents but there are {n} units."
            )
        emb = text_embedding(texts, n_components=n_components,
                             min_df=min_df, max_df=max_df)
    E = emb["embedding"]

    parts = [E]
    if X is not None:
        Xa = np.atleast_2d(np.asarray(X, dtype=float))
        if Xa.shape[0] != n:
            Xa = Xa.T
        if Xa.shape[0] != n:
            raise ValueError(f"X has {Xa.shape[0]} rows but there are {n}.")
        parts.append(Xa)
    W = np.column_stack(parts)
    Wd = add_intercept(W)

    gbeta, separated = logit_fit(Wd, t)
    e_raw = logit_predict(Wd, gbeta)
    e = np.clip(e_raw, trim, 1.0 - trim)
    n_trim = int(np.sum((e_raw < trim) | (e_raw > 1.0 - trim)))

    m1, m0 = t == 1.0, t == 0.0
    if m1.sum() < Wd.shape[1] + 1 or m0.sum() < Wd.shape[1] + 1:
        raise ValueError(
            f"an arm has too few units ({int(m1.sum())} treated, "
            f"{int(m0.sum())} control) for {Wd.shape[1]} design columns. "
            "Reduce n_components."
        )
    b1, *_ = np.linalg.lstsq(Wd[m1], y[m1], rcond=None)
    b0, *_ = np.linalg.lstsq(Wd[m0], y[m0], rcond=None)
    mu1, mu0 = Wd @ b1, Wd @ b0

    psi = mu1 - mu0 + t * (y - mu1) / e - (1 - t) * (y - mu0) / (1 - e)
    ate = float(np.mean(psi))
    se = float(math.sqrt(np.mean((psi - ate) ** 2) / n))
    naive = float(np.mean(y[m1]) - np.mean(y[m0]))

    wts = np.where(t == 1.0, 1.0 / e, 1.0 / (1.0 - e))
    share = float(np.max(wts) / np.sum(wts))

    zc = _z(1 - alpha / 2)
    evr = emb["explained_variance_ratio"]
    out = RichResult(
        title="Treatment effect with a text-derived adjustment set",
        summary_lines=[
            ("AIPW ATE", ate),
            ("SE", se),
            ("Unadjusted difference", naive),
            ("Movement from unadjusted", ate - naive),
            ("Text components", int(E.shape[1])),
        ],
        payload={
            "estimate": ate,
            "se": se,
            "ci_lower": ate - zc * se,
            "ci_upper": ate + zc * se,
            "naive_difference": naive,
            "adjustment_movement": ate - naive,
            "propensity": e,
            "propensity_untrimmed": e_raw,
            "n_trimmed": n_trim,
            "max_weight_share": share,
            "embedding": E,
            "n_components": int(E.shape[1]),
            "explained_variance_ratio": evr,
            "cumulative_variance": (float(np.nansum(evr))
                                    if evr.size else float("nan")),
            "vocabulary_size": emb["n_terms"],
            "mu1": mu1,
            "mu0": mu0,
            "separated": separated,
            "n": n,
            "n_treated": int(m1.sum()),
            "method": _METHOD,
        },
        interpretation=(
            f"Adjusting for the text moved the estimate from {naive:.4f} to "
            f"{ate:.4f}. That movement is a lower bound on the text-borne "
            "confounding, not a demonstration that none remains."
        ),
    )
    out.warnings.append(
        "The adjustment set is a bag-of-words representation, which cannot "
        "encode negation or word order. Confounding carried by those is not "
        "removed, and the estimate will look adjusted regardless."
    )
    if separated:
        out.warnings.append(
            "The propensity model separates the data: the text predicts "
            "treatment almost perfectly. There is no comparable control for "
            "some treated documents, so the effect is not identified for "
            "them however the weights are trimmed."
        )
    if share > 0.05:
        out.warnings.append(
            f"A single unit carries {share:.1%} of the total weight. The "
            "estimate rests on very few documents; reduce n_components."
        )
    if n_trim:
        out.warnings.append(
            f"{n_trim} of {n} propensity scores were trimmed to "
            f"[{trim}, {1 - trim}], so the estimand is for the trimmed "
            "subpopulation."
        )
    return out


def _z(q):
    lo, hi = -12.0, 12.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 0.5 * math.erfc(-mid / math.sqrt(2.0)) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def cheatsheet():
    return (
        "cbnrt: treatment effects adjusting for text, via a native TF-IDF "
        "and truncated SVD representation feeding an AIPW estimator"
    )


# compact alias per ledger/NAMING.md
causalberttext = causalbert_text


# compact alias per ledger/NAMING.md
textembedding = text_embedding


# compact alias per ledger/NAMING.md
tfidfmatrix = tfidf_matrix
