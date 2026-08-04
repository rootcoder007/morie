# morie.fn -- function file (rootcoder007/morie)
"""Absolute moment of a centred normal."""

import math

from ._richresult import RichResult

__all__ = ["turboquant_normal_moment"]


def turboquant_normal_moment(sigma, l):
    """The l-th absolute moment of a zero-mean normal.

    This is the fact that makes the sign quantizer work.  The estimator
    turns an inner product into ``E|x|`` for a Gaussian ``x`` with
    variance ``||k||^2``, and the first absolute moment is
    ``sigma sqrt(2 / pi)`` -- which is exactly why the ``sqrt(pi / 2)``
    appears out front of the estimator, cancelling it.

    Formula: ``E|X|^l = sigma^l 2^(l/2) Gamma((l + 1) / 2) / sqrt(pi)``.

    Parameters
    ----------
    sigma : float
        Standard deviation.
    l : float
        Moment order; need not be an integer.

    Returns
    -------
    RichResult
        ``moment``, ``estimate`` (the same value), ``sigma``, ``l``.

    References
    ----------
    Zandieh, A., Daliri, M. & Han, I. (2024).  QJL: 1-bit quantized
    JL transform for KV cache quantization with zero overhead.
    arXiv:2406.03482.  Fetched and read; the definitions and bounds used
    here are that paper own (definition 3.1, fact 3.4, lemma 3.5,
    theorem 3.6).  The KV-cache system built on it is Zandieh, A. et al.
    (2025), TurboQuant: online vector quantization with near-optimal
    distortion rate, arXiv:2504.19874.
    """
    sigma = float(sigma)
    l = float(l)
    val = sigma ** l * 2.0 ** (l / 2.0) * math.gamma((l + 1.0) / 2.0) / math.sqrt(math.pi)
    return RichResult(payload={
        "moment": val, "estimate": val, "sigma": sigma, "l": l,
        "method": "Absolute moment of a centred normal"})


def cheatsheet():
    return "tqmom: Absolute moment of a centred normal."
