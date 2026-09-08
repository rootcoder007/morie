# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Perplexity of a model on a token sequence."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_perplexity"]


def kamath_perplexity(log_probs, base="e"):
    """PPL = exp(-(1/N) sum_t log p_theta(x_t | x_{<t})).

    ``log_probs`` are NATURAL log-probabilities of the observed
    tokens, so every entry must be <= 0; a positive entry means
    probabilities or logits were passed instead, which would silently
    return a perplexity below 1 -- impossible, and the single most
    common way this number is reported wrong. A log-probability of
    -inf gives infinite perplexity, which is the mathematics and is
    returned, not clipped.

    Reference: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 8, perplexity.

    Examples
    --------
    >>> import math
    >>> out = kamath_perplexity([-math.log(2), -math.log(2)])
    >>> abs(out["estimate"] - 2.0) < 1e-12
    True
    >>> out["mean_nll"] > 0
    True
    >>> kamath_perplexity([0.0, 0.0])["estimate"]
    1.0
    """
    lp = np.atleast_1d(np.asarray(log_probs, dtype=float)).ravel()
    if lp.size == 0:
        raise ValueError(
            "no tokens scored; perplexity over an empty sequence is "
            "undefined, not 1.")
    if np.any(np.isnan(lp)):
        raise ValueError("a log-probability is nan.")
    if np.any(lp > 0):
        bad = float(lp[lp > 0][0])
        raise ValueError(
            f"log-probabilities must be <= 0; got {bad}. These look "
            "like probabilities or logits, not natural logs.")
    if base not in ("e", "2"):
        raise ValueError("base must be 'e' (nats) or '2' (bits).")
    mean_nll = float(-lp.mean())
    ppl = float(np.exp(mean_nll)) if base == "e" else float(2.0 ** mean_nll)
    return RichResult(payload={
        "estimate": ppl, "perplexity": ppl,
        "mean_nll": mean_nll,
        "total_nll": float(-lp.sum()),
        "bits_per_token": mean_nll / np.log(2.0),
        "base": base, "n": int(lp.size),
        "method": "Perplexity exp(mean negative log-likelihood)"})


def cheatsheet():
    return "kmperp: exp(-mean log p); positive log-probs refused"
