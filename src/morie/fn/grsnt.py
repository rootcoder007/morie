# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Binary sentiment classification over pooled token embeddings."""

from . import _array_core as np

from ._richresult import RichResult
from .grsig import geron_sigmoid

__all__ = ["geron_sentiment_binary"]

_METHOD = "Binary sentiment head over pooled embeddings"


def geron_sentiment_binary(token_ids, E, w, b=0.0, pooling="mean", threshold=0.5):
    r"""Pool a sequence of embeddings and score it with one logistic unit.

    .. math::
        p = \sigma\bigl(\mathbf{w}^{\top}\mathrm{pool}(E[x]) + b\bigr)

    Mean pooling throws word order away entirely, which is why this
    bag-of-embeddings baseline is surprisingly hard to beat on sentiment
    (the signal is largely lexical) and hopeless on negation ("not good"
    pools to roughly the average of "not" and "good").  Max pooling picks
    the strongest feature per dimension instead, which behaves like a
    keyword detector.  Both are offered because the choice, not the
    classifier, is what decides the failure mode.  The sigmoid is
    delegated to :mod:`morie.fn.grsig`.

    Parameters
    ----------
    token_ids : sequence of int
        Non-empty; every id must index a row of ``E``.
    E : array-like, shape (V, d)
        Embedding matrix.
    w : array-like, shape (d,)
    b : float, optional
    pooling : {"mean", "max", "sum"}, optional
    threshold : float, optional
        Decision threshold in ``[0, 1]``.

    Returns
    -------
    RichResult
        Payload keys ``probability``, ``label`` (1 = positive),
        ``logit``, ``pooled``, ``token_contributions``, ``estimate``,
        ``n``, ``method``.

    References
    ----------
    Géron Ch 14, Sentiment Analysis section.

    Examples
    --------
    Two tokens whose embeddings mean to ``[1.5]``, weight 1, bias 0:
    the logit is 1.5 and ``sigma(1.5) = 0.817574``.

    >>> E = [[1.0], [2.0]]
    >>> r = geron_sentiment_binary([0, 1], E, [1.0])
    >>> r["pooled"], round(r["logit"], 6)
    ([1.5], 1.5)
    >>> round(r["probability"], 6)
    0.817574
    >>> r["label"]
    1

    Max pooling keeps the strongest token instead of averaging it away:

    >>> geron_sentiment_binary([0, 1], E, [1.0], pooling="max")["pooled"]
    [2.0]
    """
    ids = np.asarray(token_ids).ravel()
    if ids.size == 0:
        raise ValueError("token_ids is empty; there is nothing to pool.")
    if not np.all(ids == np.round(np.asarray(ids, dtype=float))):
        raise ValueError("token_ids must be integers.")
    ids = ids.astype(int)
    Em = np.atleast_2d(np.asarray(E, dtype=float))
    if Em.ndim != 2 or Em.size == 0:
        raise ValueError(f"E must be a non-empty (V, d) matrix, got shape {Em.shape}.")
    if ids.min() < 0 or ids.max() >= Em.shape[0]:
        raise ValueError(
            f"token ids must lie in [0, {Em.shape[0] - 1}], got "
            f"[{int(ids.min())}, {int(ids.max())}]."
        )
    wv = np.asarray(w, dtype=float).ravel()
    if wv.size != Em.shape[1]:
        raise ValueError(f"w has {wv.size} weights but embeddings are {Em.shape[1]}-dimensional.")
    if not np.all(np.isfinite(Em)) or not np.all(np.isfinite(wv)):
        raise ValueError("E and w must be finite.")
    b = float(b)
    if not np.isfinite(b):
        raise ValueError(f"b must be finite, got {b}.")
    threshold = float(threshold)
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must lie in [0, 1], got {threshold}.")

    V = Em[ids]
    if pooling == "mean":
        pooled = V.mean(axis=0)
    elif pooling == "max":
        pooled = V.max(axis=0)
    elif pooling == "sum":
        pooled = V.sum(axis=0)
    else:
        raise ValueError(f"pooling must be 'mean', 'max' or 'sum', got {pooling!r}.")

    logit = float(pooled @ wv + b)
    p = float(geron_sigmoid(logit)["sigma"])

    return RichResult(
        title="Binary sentiment",
        summary_lines=[("p(positive)", p), ("Tokens", int(ids.size)), ("Pooling", pooling)],
        payload={
            "probability": p,
            "label": int(p >= threshold),
            "logit": logit,
            "pooled": pooled.tolist(),
            "token_contributions": (V @ wv).tolist(),
            "estimate": p,
            "n": int(ids.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grsnt: p = sigmoid(w . pool(E[ids]) + b); mean pooling loses order, max acts as keyword detector"
