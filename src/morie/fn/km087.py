# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 6.11: the CrowS-Pairs Score."""

from . import _array_core as np

from ._richresult import RichResult
from .km086 import kamath_ch6_pll

__all__ = ["kamath_ch6_cps_metric"]


def kamath_ch6_cps_metric(U, M, theta=None):
    """CPS(S) = sum_{u in U} log P(u | U_without_u, M ; theta).

    PLL restricted to the UNMODIFIED tokens U and conditioned on the
    modified ones M (the protected attributes), so the score measures
    the shared sentence under a swapped demographic rather than the
    demographic word itself. The sum is Eq 6.10's, delegated to km086.
    ``theta`` is a callable (U, M, i) -> probability; with None, ``U``
    holds those probabilities already.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 6, Eq 6.11, printed
    p. 236.

    Examples
    --------
    >>> import math
    >>> out = kamath_ch6_cps_metric([0.5, 0.5], ["he"])
    >>> abs(out["estimate"] + math.log(4.0)) < 1e-12
    True
    >>> kamath_ch6_cps_metric(["x"], ["he"],
    ...                       theta=lambda U, M, i: 0.5)["n_modified"]
    1
    """
    toks = list(U)
    mod = list(M)
    if not mod:
        raise ValueError("M is empty; CPS conditions on the modified "
                         "tokens, so there must be at least one.")
    if theta is None:
        probs = toks
    else:
        if not callable(theta):
            raise ValueError("theta must be a callable (U, M, i) -> "
                             "probability.")
        probs = [float(theta(toks, mod, i)) for i in range(len(toks))]
    inner = kamath_ch6_pll(probs)
    return RichResult(payload={
        "estimate": inner["estimate"], "per_token": inner["per_token"],
        "n_modified": len(mod), "n": inner["n"],
        "method": "CrowS-Pairs Score (Kamath Eq 6.11)"})


def cheatsheet():
    return "km087: PLL over the unmodified tokens, given the modified"
