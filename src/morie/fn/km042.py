# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 3.1: prompt-based classification through a label word map."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_ch3_prompt_label_mapping"]


def _validate_dist(d, name):
    if not isinstance(d, dict) or not d:
        raise ValueError(f"{name} must be a non-empty mapping "
                         f"answer-word -> probability.")
    p = np.asarray([float(v) for v in d.values()], dtype=float)
    if np.any(p < 0):
        raise ValueError(f"{name} holds a negative probability.")
    if abs(float(p.sum()) - 1.0) > 1e-8:
        raise ValueError(
            f"{name} must sum to 1; it sums to {float(p.sum()):.6g}.")
    return p


def kamath_ch3_prompt_label_mapping(x, y, M):
    """p(y|x) = p(z = M(y) | x'): the class probability IS the answer
    word's probability under the filled prompt.

    ``x`` is the model's distribution over answer words z given the
    prompted input x' (mapping word -> probability, validated to sum
    to 1); ``M`` maps class labels to answer words; ``y`` is the class
    asked about. No head, no new parameters -- that is the point of
    Eq 3.1.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 3, Eq 3.1, printed
    p. 91.

    Examples
    --------
    >>> M = {"positive": "great", "negative": "terrible"}
    >>> out = kamath_ch3_prompt_label_mapping(
    ...     {"great": 0.7, "terrible": 0.3}, "positive", M)
    >>> out["estimate"]
    0.7
    >>> out["label_probs"]["negative"]
    0.3
    """
    _validate_dist(x, "x")
    if not isinstance(M, dict) or not M:
        raise ValueError("M must be a non-empty label -> answer-word map.")
    if y not in M:
        raise ValueError(f"label {y!r} is not in the label word map M.")
    missing = [w for w in M.values() if w not in x]
    if missing:
        raise ValueError(
            f"the answer words {missing!r} carry no probability in x; "
            "the label map and the distribution disagree.")
    label_probs = {k: float(x[w]) for k, w in M.items()}
    return RichResult(payload={
        "estimate": float(x[M[y]]), "label": y, "answer_word": M[y],
        "label_probs": label_probs,
        "label_mass": float(sum(label_probs.values())),
        "n": len(x),
        "method": "prompt label word mapping (Kamath Eq 3.1)"})


def cheatsheet():
    return "km042: p(y|x) = p(z = M(y)|x'), class prob via answer word"
