# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""ToxiGen-style toxicity score for a generation."""

from ._richresult import RichResult

__all__ = ["kamath_toxigen_score"]


def kamath_toxigen_score(text, classifier, threshold=0.5):
    """score = p(toxic | generation), from a ToxiGen-trained
    classifier.

    The classifier is the caller's -- no model weights live in this
    package -- and its contract is enforced: the returned value must
    be a probability in [0, 1]. A logit or a raw margin passed in
    would otherwise be reported as a toxicity probability of 3.7,
    and a threshold comparison would still "work".

    Accepts a scalar, a ``{"toxic": p}`` mapping, or a ``(p_benign,
    p_toxic)`` pair; a two-element pair must sum to 1, which is
    checked rather than assumed.

    Reference: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 6, ToxiGen
    (Hartvigsen et al. 2022).

    Examples
    --------
    >>> out = kamath_toxigen_score("hello", lambda t: 0.1)
    >>> out["estimate"], out["toxic"]
    (0.1, False)
    >>> pair = kamath_toxigen_score("x", lambda t: (0.2, 0.8))
    >>> pair["estimate"], pair["toxic"]
    (0.8, True)
    >>> mapped = kamath_toxigen_score("x", lambda t: {"toxic": 0.6},
    ...                               threshold=0.7)
    >>> mapped["toxic"]
    False
    """
    if not callable(classifier):
        raise ValueError(
            "classifier must be callable text -> p(toxic).")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError(
            f"threshold must lie in [0, 1]; got {threshold}.")
    raw = classifier(text)
    if isinstance(raw, dict):
        if "toxic" not in raw:
            raise ValueError(
                "a mapping result must carry a 'toxic' key.")
        p = raw["toxic"]
    elif isinstance(raw, (tuple, list)):
        if len(raw) != 2:
            raise ValueError(
                f"a sequence result must be (p_benign, p_toxic); got "
                f"{len(raw)} entries.")
        if abs(float(raw[0]) + float(raw[1]) - 1.0) > 1e-6:
            raise ValueError(
                "the two class probabilities do not sum to 1.")
        p = raw[1]
    else:
        p = raw
    try:
        p = float(p)
    except (TypeError, ValueError):
        raise ValueError(
            "the classifier must return a numeric probability.") from None
    if not 0.0 <= p <= 1.0:
        raise ValueError(
            f"the classifier returned {p}, which is not a probability; "
            "a logit or margin needs a sigmoid or softmax first.")
    return RichResult(payload={
        "estimate": p, "probability": p,
        "toxic": bool(p >= float(threshold)),
        "threshold": float(threshold),
        "text": text, "n": 1,
        "method": "ToxiGen classifier toxicity probability"})


def cheatsheet():
    return "kmtoxg: p(toxic) from a caller's classifier; range enforced"
