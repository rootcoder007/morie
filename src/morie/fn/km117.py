# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 8.5: the final BLEU score."""

from ._richresult import RichResult
from .km115 import kamath_ch8_bleu_n_geom_mean

__all__ = ["kamath_ch8_bleu_final"]


def kamath_ch8_bleu_final(BP, p_n, N=None):
    r"""BLEU = BP * exp(sum_n (1/N) log p_n).

    The exponential-of-mean-log factor IS the geometric mean of Eq
    8.3, so that half is delegated to ``morie.fn.km115`` instead of
    written twice.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 8, Eq 8.5, printed
    p. 323.

    Examples
    --------
    >>> out = kamath_ch8_bleu_final(0.5, [0.5, 0.125])
    >>> round(out["estimate"], 12)      # 0.5 * 0.25
    0.125
    """
    bp = float(BP)
    if not (0.0 <= bp <= 1.0):
        raise ValueError("the brevity penalty must lie in [0, 1]; got "
                         f"{bp}.")
    gm = kamath_ch8_bleu_n_geom_mean(p_n, N)
    return RichResult(payload={
        "estimate": bp * float(gm["estimate"]),
        "brevity_penalty": bp, "geometric_mean": float(gm["estimate"]),
        "p_n": gm["p_n"], "n": gm["n"],
        "method": "BLEU (Kamath Eq 8.5; geometric mean from km115)"})


def cheatsheet():
    return "km117: BP times the km115 geometric mean of precisions"
