# morie.fn -- function file (rootcoder007/morie)
"""Perplexity of a language model."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['perplex', 'perplexity']


def perplex(log_probs, N=None):
    """Perplexity of a language model.

    Perplexity is the exponentiated cross-entropy, so it reads as an effective branching factor: a model with perplexity 100 is as uncertain as one choosing uniformly among 100 options. Both the natural-log and base-2 forms are returned because the literature uses both and the numbers are not comparable across them. N defaults to the number of supplied log-probabilities, which is right only when they are per-token; pass N explicitly when they are not.


    Formula: PPL = exp(-(1/N) sum_i log p(x_i)); cross-entropy H = -(1/N) sum_i log2 p(x_i)

    Parameters
    ----------
    log_probs : array-like
        Natural-log probabilities assigned to the observed tokens.
    N : int, optional
        Token count to normalise by; ``len(log_probs)`` if omitted.

    Returns
    -------
    RichResult
        ``perplexity``, ``cross_entropy_nats``, ``cross_entropy_bits``, ``N``.

    References
    ----------
    Brown, Della Pietra, Mercer, Della Pietra and Lai (1992), An
    estimate of an upper bound for the entropy of English,
    Computational Linguistics 18:31-40.  Not held locally; perplexity as
    the exponentiated per-token cross-entropy is the standard published
    definition.
    """
    lp = C.vec(log_probs)
    if not lp:
        raise ValueError("need at least one log-probability")
    if any(v > 0 for v in lp):
        raise ValueError("log-probabilities must be non-positive")
    n = float(N) if N is not None else float(len(lp))
    if n <= 0:
        raise ValueError("N must be positive")
    h = -sum(lp) / n
    return RichResult(payload={
        "perplexity": math.exp(h), "cross_entropy_nats": h,
        "cross_entropy_bits": h / math.log(2.0), "N": n,
        "method": "Perplexity"})


perplexity = perplex


def cheatsheet():
    return "prgxnt: Perplexity of a language model."
