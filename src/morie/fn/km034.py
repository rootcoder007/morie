# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 2.34: GPT's unsupervised objective."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_ch2_gpt_unsupervised_obj"]


def kamath_ch2_gpt_unsupervised_obj(U, k=None, Theta=None):
    """L1(U) = sum_i log P(u_i | u_i-k..u_i-1; Theta). ``U`` holds the
    model's per-token probabilities under a context of size k; the
    objective is MAXIMISED, so it is the sum of LOGS, negative for any
    imperfect model, and the payload also gives the equivalent
    per-token cross-entropy.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 2, Eq 2.34, printed
    p. 70.

    Examples
    --------
    >>> import math
    >>> abs(kamath_ch2_gpt_unsupervised_obj([0.5, 0.5])["estimate"]
    ...     + 2 * math.log(2)) < 1e-12
    True
    """
    p = np.atleast_1d(np.asarray(U, dtype=float))
    if len(p) == 0:
        raise ValueError("no token probabilities supplied.")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must lie in [0, 1].")
    if k is not None and int(k) < 1:
        raise ValueError("the context size k must be positive.")
    with np.errstate(divide="ignore"):
        logs = np.log(p)
    return RichResult(payload={
        "estimate": float(logs.sum()),
        "cross_entropy": float(-logs.mean()),
        "context_size": None if k is None else int(k), "n": len(p),
        "method": "GPT unsupervised objective L1 (Kamath Eq 2.34)"})


def cheatsheet():
    return "km034: sum of log-probs, maximised; CE reported alongside"
