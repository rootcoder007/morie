# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Text-to-text classification, T5 style (Raffel et al. 2020;
Alammar Ch 4)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["alammar_t5_text_to_text_classify"]


def alammar_t5_text_to_text_classify(input_text, label_tokens, model,
                                     prefix=""):
    """y = argmax over label strings of p_model(label | prefix + input).

    ``model`` is a callable (input_text, label) -> log-probability.
    Scores are renormalised over the CLOSED label set, which is the
    trick that turns a text generator into a classifier.

    References: Alammar and Grootendorst, Ch 4; Raffel et al. (2020).
    """
    labels = [str(l) for l in label_tokens]
    if not labels:
        raise ValueError("no label tokens supplied.")
    if not callable(model):
        raise ValueError("model must be a callable "
                         "(input, label) -> log-probability.")
    lp = np.array([float(model(prefix + str(input_text), l))
                   for l in labels])
    z = lp - lp.max()
    p = np.exp(z) / np.exp(z).sum()
    best = int(np.argmax(p))
    return RichResult(payload={
        "predicted_label": labels[best],
        "probabilities": {labels[i]: float(p[i])
                          for i in range(len(labels))},
        "log_scores": [float(v) for v in lp],
        "estimate": float(p[best]), "n": len(labels),
        "method": "T5 text-to-text classification (Raffel et al. 2020)"})


def cheatsheet():
    return "alt5c: renormalise generator log-probs over a closed label set"
