# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 9.17: the MMLLM autoregressive instruction-tuning loss."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_ch9_mmllm_autoregressive"]


def kamath_ch9_mmllm_autoregressive(R, I, theta=None):
    r"""L(theta) = -sum_{i=1..N} log p(R_i | I, R_{<i}; theta).

    ``R`` holds the model's probability of each ground-truth response
    token in context; ``theta``, if given, is a callable
    ``theta(R, I) -> those probabilities``. The sum runs over the whole
    response (N tokens), so this is a sequence log-likelihood, not a
    per-token average -- ``mean_nll`` is reported alongside for
    comparability across lengths.

    ``morie.fn.km142`` (Eq 9.14) is this same sum over a set of pairs
    and delegates here.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 9, Eq 9.17, printed
    p. 391.

    Examples
    --------
    >>> import math
    >>> out = kamath_ch9_mmllm_autoregressive([0.5, 0.25], None)
    >>> abs(out["estimate"] - math.log(8)) < 1e-12
    True
    """
    if theta is not None:
        if not callable(theta):
            raise ValueError("theta must be a callable theta(R, I) or "
                             "None when R already holds the "
                             "per-token probabilities.")
        R = theta(R, I)
    p = np.atleast_1d(np.asarray(R, dtype=float)).ravel()
    if p.size == 0:
        raise ValueError("the response is empty; its log-likelihood is "
                         "not 0 but undefined.")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("response-token probabilities must lie in "
                         "[0, 1].")
    with np.errstate(divide="ignore"):
        nll = -np.log(p)
    total = float(nll.sum())
    return RichResult(payload={
        "estimate": total, "mean_nll": total / p.size,
        "per_token": [float(u) for u in nll],
        "sequence_probability": float(np.prod(p)), "n": int(p.size),
        "method": "MMLLM autoregressive response loss (Kamath Eq 9.17)"})


def cheatsheet():
    return "km145: -sum log p over the ground-truth response tokens"
