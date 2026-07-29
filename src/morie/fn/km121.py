# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 8.9: BERTScore F1."""

from ._richresult import RichResult

__all__ = ["kamath_ch8_bertscore_f1"]


def kamath_ch8_bertscore_f1(P_BERT, R_BERT):
    r"""F_BERT = 2 * P * R / (P + R), the harmonic mean.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 8, Eq 8.9, printed
    p. 325.

    Examples
    --------
    >>> out = kamath_ch8_bertscore_f1(1.0, 0.5)
    >>> round(out["estimate"], 6)      # 2*0.5/1.5
    0.666667
    """
    p = float(P_BERT)
    r = float(R_BERT)
    if p + r == 0:
        raise ValueError("precision and recall are both 0, so the "
                         "harmonic mean is 0/0 -- undefined.")
    if p + r < 0:
        raise ValueError("a negative precision+recall has no harmonic "
                         "mean on this scale; check the inputs.")
    return RichResult(payload={
        "estimate": 2.0 * p * r / (p + r), "precision": p, "recall": r,
        "n": 2, "method": "BERTScore F1 (Kamath Eq 8.9)"})


def cheatsheet():
    return "km121: harmonic mean of BERTScore precision and recall"
