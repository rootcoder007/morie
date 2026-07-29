# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 9.21: Flamingo's factorized text likelihood."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_ch9_flamingo_factorized"]


def kamath_ch9_flamingo_factorized(y, x=None, L=None, model=None):
    r"""p(y|x) = prod_{l=1..L} p(y_l | y_<l, x_<=l).

    ``y`` holds the per-token conditional probabilities
    p(y_l | y_<l, x_<=l) -- one entry per generated token; ``model``,
    if given, is a callable ``model(y, x) -> those probabilities``.
    ``L`` is checked against the sequence length. The product is
    accumulated in log space and both forms are returned, so a long
    caption does not underflow to a bare 0.0 without its log.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 9, Eq 9.21, printed
    p. 404; Alayrac et al. (2022).

    Examples
    --------
    >>> out = kamath_ch9_flamingo_factorized([0.5, 0.25])
    >>> out["estimate"]
    0.125
    """
    if model is not None:
        if not callable(model):
            raise ValueError("model must be callable model(y, x) or "
                             "None when y already holds the per-token "
                             "conditionals.")
        y = model(y, x)
    p = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
    if p.size == 0:
        raise ValueError("the sequence is empty; p(y|x) over no tokens "
                         "is undefined.")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("conditional probabilities must lie in [0, 1].")
    if L is not None and int(L) != p.size:
        raise ValueError(
            f"L = {L} contradicts the {p.size} tokens given.")
    with np.errstate(divide="ignore"):
        logp = float(np.log(p).sum())
    return RichResult(payload={
        "estimate": float(np.prod(p)), "log_prob": logp,
        "nll": -logp, "per_token": [float(v) for v in p],
        "n": int(p.size),
        "method": "Flamingo factorized text likelihood "
                  "(Kamath Eq 9.21)"})


def cheatsheet():
    return "km149: product of per-token conditionals, with its log"
