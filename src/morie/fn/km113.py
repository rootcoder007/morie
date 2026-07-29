# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 8.1: perplexity of a tokenized sequence."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_ch8_perplexity"]


def kamath_ch8_perplexity(X, N=None, p_theta=None):
    r"""PPL(X) = exp(-(1/N) sum_i log p_theta(x_i | x_<i)).

    ``p_theta`` is either a callable ``p_theta(x_i, x_prefix) -> prob``
    scored over every token of ``X``, or a pre-computed array of the
    model's probability of each token in context; ``X`` is then the
    token sequence (used for length and for the callable). ``N``, if
    given, must equal the number of tokens scored.

    A zero probability makes the log-likelihood -inf and the
    perplexity +inf: that is the mathematics of a token the model
    ruled out, and it is returned, not clipped.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 8, Eq 8.1, printed
    p. 322.

    Examples
    --------
    >>> out = kamath_ch8_perplexity(["a", "b"], p_theta=[0.5, 0.5])
    >>> out["estimate"]
    2.0
    """
    toks = list(X) if not isinstance(X, np.ndarray) else list(X.ravel())
    if len(toks) == 0:
        raise ValueError("an empty sequence has no perplexity.")
    if p_theta is None:
        raise ValueError("p_theta is required: either a callable "
                         "p_theta(x_i, x_prefix) or one probability "
                         "per token.")
    if callable(p_theta):
        probs = np.array([float(p_theta(toks[i], toks[:i]))
                          for i in range(len(toks))], dtype=float)
    else:
        probs = np.atleast_1d(np.asarray(p_theta, dtype=float))
        if probs.size != len(toks):
            raise ValueError(
                f"{probs.size} probabilities for {len(toks)} tokens.")
    if np.any((probs < 0) | (probs > 1)):
        raise ValueError("token probabilities must lie in [0, 1].")
    if N is not None and int(N) != probs.size:
        raise ValueError(
            f"N = {N} contradicts the {probs.size} tokens scored.")
    with np.errstate(divide="ignore"):
        logp = np.log(probs)
    nll = float(-logp.mean())
    return RichResult(payload={
        "estimate": float(np.exp(nll)), "mean_nll": nll,
        "log_probs": [float(v) for v in logp], "n": int(probs.size),
        "method": "perplexity (Kamath Eq 8.1)"})


def cheatsheet():
    return "km113: exp(mean negative log-likelihood per token)"
