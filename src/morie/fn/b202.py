# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Burkov's Eq 2.2: the autoregressive next-token probability,
made operational as a bigram MLE over the supplied sequence."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["burkov_lm_ch2_lm_next_token"]


def burkov_lm_ch2_lm_next_token(t_next, s):
    """Pr(t = t_next | s), estimated by bigram MLE from s itself.

    Eq 2.2 DEFINES a language model rather than giving an estimator,
    so the operational content here is the simplest one the book then
    builds on: count how often the last token of s is followed by
    t_next within s, over how often it is followed by anything.

    References: Burkov LM (2025), Ch 2, Eq 2.2, p. 76.

    Examples
    --------
    >>> burkov_lm_ch2_lm_next_token("b", ["a", "b", "a", "b", "a"])["estimate"]
    1.0
    """
    seq = [str(t) for t in np.atleast_1d(np.asarray(s, dtype=object))]
    if len(seq) < 2:
        raise ValueError("need at least 2 tokens to form one bigram.")
    t_next = str(t_next)
    ctx = seq[-1]
    follow = [seq[i + 1] for i in range(len(seq) - 1) if seq[i] == ctx]
    if not follow:
        raise ValueError(
            f"the context token {ctx!r} never has a successor in s, so "
            "the MLE conditional is undefined (0/0).")
    p = follow.count(t_next) / len(follow)
    dist = {t: follow.count(t) / len(follow) for t in sorted(set(follow))}
    return RichResult(payload={
        "estimate": float(p), "context": ctx, "distribution": dist,
        "n": len(seq),
        "method": "Autoregressive next-token probability, bigram MLE "
                  "(Burkov Eq 2.2)"})


def cheatsheet():
    return "b202: next-token probability Pr(t | s), bigram MLE (Burkov Eq 2.2)"
