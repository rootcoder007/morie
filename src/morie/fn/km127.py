# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 8.15: the G-Eval probability-weighted score."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_ch8_geval_score"]


def kamath_ch8_geval_score(s_i, p):
    r"""score = sum_i p(s_i) * s_i.

    ``s_i`` are the discrete scores the evaluation prompt allows and
    ``p`` the evaluator LLM's probability for each. The probabilities
    must form a distribution over those scores (sum 1 to 1e-6);
    G-Eval's whole point is that the expectation de-quantizes the
    integer rating, so a non-distribution is an error, not a
    rescaling.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 8, Eq 8.15, printed
    p. 328.

    Examples
    --------
    >>> out = kamath_ch8_geval_score([1, 2, 3], [0.2, 0.3, 0.5])
    >>> round(out["estimate"], 12)     # 0.2 + 0.6 + 1.5
    2.3
    """
    s = np.atleast_1d(np.asarray(s_i, dtype=float))
    q = np.atleast_1d(np.asarray(p, dtype=float))
    if s.size == 0:
        raise ValueError("no candidate scores were given.")
    if s.shape != q.shape:
        raise ValueError(
            f"{q.size} probabilities for {s.size} scores.")
    if np.any(q < 0):
        raise ValueError("score probabilities cannot be negative.")
    if abs(float(q.sum()) - 1.0) > 1e-6:
        raise ValueError(
            f"the score probabilities sum to {q.sum()}, not 1; G-Eval "
            "needs a distribution over the allowed scores.")
    return RichResult(payload={
        "estimate": float(np.dot(q, s)),
        "scores": [float(v) for v in s],
        "probabilities": [float(v) for v in q], "n": int(s.size),
        "method": "G-Eval probability-weighted score (Kamath Eq 8.15)"})


def cheatsheet():
    return "km127: expected rating under the evaluator's score posterior"
