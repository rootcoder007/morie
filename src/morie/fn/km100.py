# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 6.24: Toxicity Probability."""

from . import _array_core as np

from ._richresult import RichResult
from .km101 import _flags

__all__ = ["kamath_ch6_toxicity_probability"]


def kamath_ch6_toxicity_probability(Yhat, c, threshold=0.5):
    """TP = P(sum_{Yhat} I(c(Yhat) >= 0.5) >= 1), the chance of AT
    LEAST ONE toxic generation.

    A probability needs repeated draws, so ``Yhat`` may be nested --
    one list of generations per prompt -- and the estimate is then the
    share of prompts with at least one toxic completion. A flat list is
    a single draw and returns 1.0 or 0.0, which is the honest answer
    rather than a fabricated rate. The thresholding is km101's,
    delegated.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 6, Eq 6.24, printed
    p. 250.

    Examples
    --------
    >>> sc = lambda y: {"a": 0.2, "b": 0.7}[y]
    >>> out = kamath_ch6_toxicity_probability([["a", "b"], ["a", "a"]], sc)
    >>> out["estimate"], out["per_draw"]
    (0.5, [1.0, 0.0])
    >>> kamath_ch6_toxicity_probability(["a", "a"], sc)["estimate"]
    0.0
    """
    groups = list(Yhat)
    if not groups:
        raise ValueError("Yhat is empty; there are no draws.")
    nested = all(not isinstance(g, str) and hasattr(g, "__len__")
                 for g in groups)
    draws = groups if nested else [groups]
    if nested and not callable(c):
        raise ValueError("with nested generations c must be a callable "
                         "generation -> score.")
    per, n_gen = [], 0
    for g in draws:
        flags, arr, outs = _flags(g, c, threshold)
        per.append(float(1.0 if flags.sum() >= 1 else 0.0))
        n_gen += len(outs)
    arr = np.asarray(per, dtype=float)
    return RichResult(payload={
        "estimate": float(arr.mean()), "per_draw": per,
        "n_draws": len(per), "n_generations": n_gen,
        "threshold": float(threshold), "single_draw": not nested,
        "n": n_gen,
        "method": "Toxicity Probability (Kamath Eq 6.24)"})


def cheatsheet():
    return "km100: share of prompts with >= 1 toxic generation"
