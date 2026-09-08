# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Next sentence prediction (BERT pretraining head)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_next_sentence_prediction"]


def _lexical_encoder(tokens, segments):
    """Jaccard-overlap baseline: [overlap, length ratio, 1]. NOT a transformer."""
    seg = np.asarray(segments)
    a = {t for t, s in zip(tokens, seg) if s == 0 and t not in ("[CLS]", "[SEP]")}
    b = {t for t, s in zip(tokens, seg) if s == 1 and t not in ("[CLS]", "[SEP]")}
    union = a | b
    overlap = len(a & b) / len(union) if union else 0.0
    ratio = min(len(a), len(b)) / max(len(a), len(b)) if a and b else 0.0
    return np.array([overlap, ratio, 1.0])


def geron_next_sentence_prediction(sent_A, sent_B, encoder=None, w=None, b=0.0, label=None):
    """
    Next sentence prediction pretraining (BERT).

    Formula: binary classifier on [CLS]: is sentence B the next of A?

    The input is assembled the way BERT requires -- ``[CLS] A [SEP] B
    [SEP]`` with a segment id marking which sentence each token belongs
    to -- and a linear head reads the [CLS] position alone. That is the
    whole task: the pooled [CLS] vector has to carry a RELATIONSHIP
    between two spans, not just token identities, which is what makes it
    useful for downstream entailment and question answering.

    NSP is also the part of BERT that did not survive: RoBERTa dropped it
    with no loss, on the argument that the negatives (a random sentence
    from another document) are separable by TOPIC and so teach topic
    matching rather than coherence. The default encoder here makes that
    critique concrete -- it is a Jaccard word-overlap baseline, not a
    transformer, and it does well on exactly the easy negatives NSP
    generates. Pass your own ``encoder(tokens, segments) -> h`` for the
    real thing.

    Parameters
    ----------
    sent_A, sent_B : sequence of str
        Tokenised sentences.
    encoder : callable, optional
        ``encoder(tokens, segments) -> h`` for the [CLS] position.
    w : array-like, optional
        Head weights matching ``h``; defaults to the baseline head.
    b : float, default 0.0
        Head bias.
    label : int, optional
        1 if B truly follows A; enables the loss.

    Returns
    -------
    result : RichResult
        Keys: tokens, segment_ids, logit, probability, prediction, loss,
        estimate, n, method.

    Examples
    --------
    The input is assembled with segment ids:

    >>> r = geron_next_sentence_prediction(["the", "cat", "sat"], ["the", "cat", "sat"])
    >>> r["tokens"]
    ['[CLS]', 'the', 'cat', 'sat', '[SEP]', 'the', 'cat', 'sat', '[SEP]']
    >>> [int(s) for s in r["segment_ids"]]
    [0, 0, 0, 0, 0, 1, 1, 1, 1]

    Identical sentences give overlap 1, so the baseline logit is
    4*1 + 0 - 2 = 2 and the probability is sigmoid(2):

    >>> round(float(r["logit"]), 6), round(float(r["probability"]), 6)
    (2.0, 0.880797)
    >>> int(r["prediction"])
    1

    An unrelated sentence has no overlap, so the logit is -2:

    >>> u = geron_next_sentence_prediction(["the", "cat"], ["quantum", "foam"])
    >>> round(float(u["logit"]), 6), int(u["prediction"])
    (-2.0, 0)

    With a label the binary cross-entropy is reported:

    >>> round(float(geron_next_sentence_prediction(["the", "cat"], ["quantum", "foam"],
    ...                                            label=0)["loss"]), 6)
    0.126928

    References
    ----------
    Geron Ch 15
    """
    A = list(sent_A)
    B = list(sent_B)
    if not A or not B:
        raise ValueError("geron_next_sentence_prediction: both sentences must be non-empty")
    tokens = ["[CLS]"] + [str(t) for t in A] + ["[SEP]"] + [str(t) for t in B] + ["[SEP]"]
    segments = np.array([0] * (len(A) + 2) + [1] * (len(B) + 1), dtype=int)

    enc = _lexical_encoder if encoder is None else encoder
    if not callable(enc):
        raise ValueError("geron_next_sentence_prediction: encoder must be callable")
    h = np.atleast_1d(np.asarray(enc(tokens, segments), dtype=float)).ravel()
    if h.size == 0:
        raise ValueError("geron_next_sentence_prediction: the encoder returned an empty [CLS] vector")
    if not np.all(np.isfinite(h)):
        raise ValueError("geron_next_sentence_prediction: the encoder returned non-finite values")

    if w is None:
        if encoder is not None:
            raise ValueError(
                "geron_next_sentence_prediction: a custom encoder needs its own head weights w "
                f"(the default head expects the 3-vector of the lexical baseline, got {h.size} features)"
            )
        wv = np.array([4.0, 0.0, -2.0])
    else:
        wv = np.atleast_1d(np.asarray(w, dtype=float)).ravel()
    if wv.size != h.size:
        raise ValueError(f"geron_next_sentence_prediction: w has {wv.size} entries but the [CLS] vector has {h.size}")

    logit = float(h @ wv + float(b))
    prob = float(1.0 / (1.0 + np.exp(-logit)))
    pred = int(prob >= 0.5)
    loss = None
    if label is not None:
        y = int(label)
        if y not in (0, 1):
            raise ValueError(f"geron_next_sentence_prediction: label must be 0 or 1, got {label!r}")
        p = min(max(prob, 1e-15), 1 - 1e-15)
        loss = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)))

    return RichResult(
        title="Next sentence prediction",
        summary_lines=[("P(is next)", prob), ("Prediction", pred), ("Tokens", len(tokens))],
        warnings=(
            ["the default encoder is a word-overlap baseline, not a transformer; it illustrates why NSP is easy"]
            if encoder is None
            else []
        ),
        interpretation="RoBERTa dropped NSP: random negatives differ by topic, so the task rewards topic matching.",
        payload={
            "tokens": tokens,
            "segment_ids": segments,
            "cls_vector": h,
            "logit": logit,
            "probability": prob,
            "prediction": pred,
            "loss": loss,
            "estimate": prob,
            "n": len(tokens),
            "method": "NSP input assembly with a linear [CLS] head",
        },
    )


def cheatsheet():
    return "hmnsp: Next sentence prediction head over [CLS]"
