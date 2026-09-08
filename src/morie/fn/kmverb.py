# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Verbalizer: map class labels to answer tokens and aggregate their
probabilities."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_verbalizer_mapping"]


def kamath_verbalizer_mapping(logits, vocab, verbalizer_map):
    """P(y_c | x) = sum over v in V_c of P_LLM(v | x).

    The sum is over the FULL vocabulary softmax, so the class
    probabilities do not add to 1 -- the mass on every token that is
    not a label word is missing, and that leftover is reported as
    ``mass_outside``. The renormalised version (the one used to
    predict) is returned separately rather than silently substituted:
    a verbalizer whose label words carry 2% of the mass predicts
    confidently from almost nothing, and only ``mass_outside`` shows
    it.

    Overlapping label sets are refused -- a token verbalising two
    classes makes the classes non-exclusive.

    Reference: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 3, verbalizer mapping.

    Examples
    --------
    >>> out = kamath_verbalizer_mapping([0.0, 0.0, 0.0, 0.0],
    ...     ["a", "b", "c", "d"], {"pos": ["a", "b"], "neg": ["c"]})
    >>> out["probabilities"]["pos"], out["probabilities"]["neg"]
    (0.5, 0.25)
    >>> out["prediction"]
    'pos'
    >>> abs(out["normalized"]["pos"] - 2 / 3) < 1e-12
    True
    >>> out["mass_outside"]
    0.25
    """
    z = np.atleast_1d(np.asarray(logits, dtype=float)).ravel()
    vocab = list(vocab)
    if z.size != len(vocab):
        raise ValueError(
            f"{z.size} logits for a vocabulary of {len(vocab)} tokens.")
    if not np.all(np.isfinite(z)):
        raise ValueError("logits must be finite.")
    if not isinstance(verbalizer_map, dict) or not verbalizer_map:
        raise ValueError(
            "verbalizer_map must be a non-empty dict "
            "{class: [answer tokens]}.")
    index = {t: i for i, t in enumerate(vocab)}
    if len(index) != len(vocab):
        raise ValueError("the vocabulary contains a duplicate token.")
    s = z - z.max()
    e = np.exp(s)
    p = e / e.sum()

    seen = {}
    probs, used = {}, {}
    for cls, toks in verbalizer_map.items():
        toks = list(toks)
        if not toks:
            raise ValueError(f"class {cls!r} has no answer tokens.")
        tot = 0.0
        for t in toks:
            if t not in index:
                raise ValueError(
                    f"the answer token {t!r} for class {cls!r} is not in "
                    "the vocabulary.")
            if t in seen:
                raise ValueError(
                    f"{t!r} verbalises both {seen[t]!r} and {cls!r}; the "
                    "classes would not be exclusive.")
            seen[t] = cls
            tot += float(p[index[t]])
        probs[cls] = tot
        used[cls] = toks
    total = sum(probs.values())
    if total <= 0:
        raise ValueError(
            "every answer token has probability 0; there is nothing to "
            "renormalise.")
    normalized = {c: v / total for c, v in probs.items()}
    pred = max(probs, key=lambda c: (probs[c], repr(c)))
    return RichResult(payload={
        "probabilities": probs, "normalized": normalized,
        "prediction": pred,
        "mass_on_labels": total,
        "mass_outside": 1.0 - total,
        "answer_tokens": used,
        "estimate": probs[pred],
        "n": len(probs),
        "method": "Verbalizer class probability = sum of answer-token "
                  "probabilities"})


def cheatsheet():
    return "kmverb: sum P(v) per class; leftover mass reported, not hidden"
