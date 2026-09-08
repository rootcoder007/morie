# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Zero-shot classification via entailment scores (Yin et al. 2019;
Alammar Ch 4)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["alammar_zero_shot_classification"]


def alammar_zero_shot_classification(text, candidate_labels, nli_model,
                                     hypothesis_template="This example "
                                     "is about {}."):
    """p(label | text) = softmax over the entailment scores of
    "text entails hypothesis(label)".

    ``nli_model`` is a callable (premise, hypothesis) -> entailment
    score; the hypothesis construction, softmax and argmax -- the
    algorithm of Yin et al. -- are computed here.

    References: Alammar and Grootendorst, Ch 4; Yin, Hay and Roth
    (2019).
    """
    labels = [str(l) for l in candidate_labels]
    if not labels:
        raise ValueError("no candidate labels supplied.")
    if len(set(labels)) != len(labels):
        raise ValueError("candidate labels contain duplicates.")
    if not callable(nli_model):
        raise ValueError("nli_model must be a callable "
                         "(premise, hypothesis) -> score.")
    scores = np.array([float(nli_model(str(text),
                                       hypothesis_template.format(l)))
                       for l in labels])
    z = scores - scores.max()
    p = np.exp(z) / np.exp(z).sum()
    order = np.argsort(-p)
    return RichResult(payload={
        "probabilities": {labels[i]: float(p[i]) for i in range(len(labels))},
        "predicted_label": labels[int(order[0])],
        "entailment_scores": [float(s) for s in scores],
        "estimate": float(p[int(order[0])]), "n": len(labels),
        "method": "Zero-shot NLI classification (Yin et al. 2019)"})


def cheatsheet():
    return "alzsc: softmax over entailment(text, hypothesis(label))"
