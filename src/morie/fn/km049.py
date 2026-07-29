# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 3.8: top-1 accuracy as the prompt selection metric."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_ch3_top1_prompt_metric"]


def kamath_ch3_top1_prompt_metric(R, t, P_LM):
    """A(t) = (1/|R|) sum_{(x,y) in R} delta(y = argmax_y' P_LM(y'|x,t)).

    ``R`` is the labelled set of (x, y) pairs, ``t`` the template being
    scored, ``P_LM`` a callable (x, t) -> {label: probability}
    validated to sum to 1. delta is Kronecker's delta, so this is plain
    top-1 accuracy of the template -- a proportion in [0, 1], never a
    mean of the inputs.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 3, Eq 3.8, printed
    p. 102.

    Examples
    --------
    >>> P = lambda x, t: {"pos": 0.9, "neg": 0.1}
    >>> out = kamath_ch3_top1_prompt_metric(
    ...     [("a", "pos"), ("b", "neg")], "T1", P)
    >>> out["estimate"], out["n_correct"]
    (0.5, 1)
    """
    pairs = list(R)
    if not pairs:
        raise ValueError("R is empty; accuracy over no examples is "
                         "undefined, not 0.")
    if not callable(P_LM):
        raise ValueError("P_LM must be a callable (x, t) -> "
                         "{label: probability}.")
    hits = []
    for pair in pairs:
        x, y = pair
        dist = P_LM(x, t)
        if not isinstance(dist, dict) or not dist:
            raise ValueError("P_LM must return a non-empty "
                             "{label: probability} mapping.")
        p = np.asarray([float(v) for v in dist.values()], dtype=float)
        if np.any(p < 0) or abs(float(p.sum()) - 1.0) > 1e-8:
            raise ValueError(
                "P_LM's distribution must be non-negative and sum to 1; "
                f"it sums to {float(p.sum()):.6g}.")
        if y not in dist:
            raise ValueError(f"the gold label {y!r} is absent from "
                             "P_LM's distribution.")
        top = list(dist.keys())[int(np.argmax(p))]
        hits.append(1 if top == y else 0)
    h = np.asarray(hits, dtype=float)
    return RichResult(payload={
        "estimate": float(h.mean()), "n_correct": int(h.sum()),
        "correct": [int(v) for v in h], "template": t, "n": len(pairs),
        "method": "top-1 prompt selection accuracy (Kamath Eq 3.8)"})


def cheatsheet():
    return "km049: top-1 accuracy of a template over a labelled set"
