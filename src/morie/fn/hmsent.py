# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Sentiment analysis with RNN or transformer on tokens."""

from . import _array_core as np

from ._richresult import RichResult
from .hmsftm import geron_softmax_function

__all__ = ["geron_sentiment_analysis"]


def geron_sentiment_analysis(texts, model, tokenizer=None, y_true=None, labels=None):
    """
    Sentiment analysis with RNN or transformer on tokens.

    Formula: y_hat = softmax(W h_T) for 2 or more sentiment classes

    Orchestrates the tokenise -> encode -> classify pipeline around a
    caller-supplied model, with the contract enforced: `tokenizer` maps a
    text to a token sequence (default: whitespace split, lowercased) and
    ``model(tokens)`` returns one score per sentiment class, the same
    number of classes for every text. Scores are normalised with
    :func:`morie.fn.hmsftm.geron_softmax_function`. When `y_true` is
    given the evaluation is computed by counting, not estimated: accuracy,
    a confusion matrix, and per-class precision / recall / F1 with their
    macro average.

    Parameters
    ----------
    texts : sequence
        Documents to classify (non-empty).
    model : callable
        ``model(tokens) -> scores`` of constant length K >= 2.
    tokenizer : callable, optional
        ``tokenizer(text) -> tokens``; default ``str.lower().split()``.
    y_true : array-like of int, optional
        Gold labels in ``0..K-1``.
    labels : sequence of str, optional
        Human-readable class names; must have length K.

    Returns
    -------
    result : RichResult
        Keys: probabilities, predicted, accuracy, confusion, precision,
        recall, f1, macro_f1, estimate, n, method.

    Examples
    --------
    A lexicon model that counts positive minus negative words:

    >>> pos, neg = {"good", "great"}, {"bad", "awful"}
    >>> def m(toks):
    ...     s = sum(t in pos for t in toks) - sum(t in neg for t in toks)
    ...     return [-s, s]
    >>> r = geron_sentiment_analysis(["good great", "awful"], m, y_true=[1, 0])
    >>> [int(v) for v in r["predicted"]]
    [1, 0]
    >>> float(r["accuracy"])
    1.0
    >>> [round(float(v), 6) for v in r["probabilities"][1]]
    [0.880797, 0.119203]

    References
    ----------
    Géron Ch 14
    """
    docs = list(texts)
    if not docs:
        raise ValueError("geron_sentiment_analysis: texts is empty")
    if not callable(model):
        raise ValueError("geron_sentiment_analysis: model must be a callable mapping tokens to class scores")
    tok = tokenizer if tokenizer is not None else (lambda t: str(t).lower().split())
    if not callable(tok):
        raise ValueError("geron_sentiment_analysis: tokenizer must be callable")

    token_lists = []
    rows = []
    K = None
    for i, doc in enumerate(docs):
        toks = list(tok(doc))
        token_lists.append(toks)
        sc = np.asarray(model(toks), dtype=float).ravel()
        if sc.size < 2:
            raise ValueError(f"geron_sentiment_analysis: model returned {sc.size} scores for text {i}; need >= 2 classes")
        if not np.all(np.isfinite(sc)):
            raise ValueError(f"geron_sentiment_analysis: model returned non-finite scores for text {i}")
        if K is None:
            K = sc.size
        elif sc.size != K:
            raise ValueError(
                f"geron_sentiment_analysis: model returned {sc.size} classes for text {i} but {K} for text 0"
            )
        rows.append(np.asarray(geron_softmax_function(sc)["p"], dtype=float))
    P = np.vstack(rows)
    pred = np.argmax(P, axis=1)

    names = None
    if labels is not None:
        names = list(labels)
        if len(names) != K:
            raise ValueError(f"geron_sentiment_analysis: {len(names)} labels supplied but the model has {K} classes")

    acc = conf = prec = rec = f1 = macro = None
    if y_true is not None:
        g = np.asarray(y_true).ravel().astype(int)
        if g.size != len(docs):
            raise ValueError(f"geron_sentiment_analysis: {len(docs)} texts but {g.size} labels")
        if g.min() < 0 or g.max() >= K:
            raise ValueError(f"geron_sentiment_analysis: y_true must lie in 0..{K - 1}, got {g.min()}..{g.max()}")
        conf = np.zeros((K, K), dtype=int)
        for a, b in zip(g, pred):
            conf[a, b] += 1
        tp = np.diag(conf).astype(float)
        prec = np.where(conf.sum(axis=0) > 0, tp / np.maximum(conf.sum(axis=0), 1), 0.0)
        rec = np.where(conf.sum(axis=1) > 0, tp / np.maximum(conf.sum(axis=1), 1), 0.0)
        denom = prec + rec
        f1 = np.where(denom > 0, 2 * prec * rec / np.where(denom > 0, denom, 1.0), 0.0)
        macro = float(np.mean(f1))
        acc = float(np.sum(tp) / conf.sum())

    return RichResult(
        title="Sentiment analysis",
        summary_lines=[
            ("Documents", len(docs)),
            ("Classes", int(K)),
            ("Accuracy", acc if acc is not None else "n/a (no labels)"),
        ],
        interpretation=(
            "Softmax over the final hidden state gives calibrated-looking probabilities, but on "
            "imbalanced sentiment data macro F1 is the number that moves when the minority class breaks."
        ),
        payload={
            "probabilities": P,
            "predicted": pred,
            "tokens": token_lists,
            "labels": names,
            "accuracy": acc,
            "confusion": conf,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "macro_f1": macro,
            "n_classes": int(K),
            "estimate": float(acc) if acc is not None else float(np.mean(np.max(P, axis=1))),
            "n": int(len(docs)),
            "method": "Tokenise -> model scores -> softmax, with counted accuracy / confusion / macro-F1",
        },
    )


def cheatsheet():
    return "hmsent: Sentiment analysis with RNN or transformer on tokens"
