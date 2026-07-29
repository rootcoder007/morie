# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 6.6: the WEAT effect size."""

import numpy as np

from ._richresult import RichResult
from .km080 import _sums

__all__ = ["kamath_ch6_weat_effect_size"]


def kamath_ch6_weat_effect_size(A_1, A_2, W_1, W_2, ddof=0):
    """WEAT = (mean_{A_1} s - mean_{A_2} s) / std_{a in A_1 u A_2} s.

    The standardised version of Eq 6.4: a Cohen's-d-shaped quantity, so
    it can be compared across tests where the raw sum cannot. The book
    does not state whether the union's standard deviation is the
    population or the sample one; ``ddof`` exposes the choice and
    defaults to 0 (population), matching the usual WEAT
    implementations. A union with no spread has no effect size and
    raises rather than dividing by zero.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 6, Eq 6.6, printed
    p. 234.

    Examples
    --------
    >>> out = kamath_ch6_weat_effect_size([[1.0, 0.0]], [[0.0, 1.0]],
    ...                                   [[1.0, 0.0]], [[0.0, 1.0]])
    >>> out["estimate"], out["std"]
    (2.0, 1.0)
    """
    s1, s2 = _sums(A_1, A_2, W_1, W_2)
    union = np.asarray(s1 + s2, dtype=float)
    ddof = int(ddof)
    if union.size - ddof <= 0:
        raise ValueError(
            f"the union holds {union.size} attribute words, too few for "
            f"ddof = {ddof}.")
    sd = float(np.std(union, ddof=ddof))
    if sd == 0:
        raise ValueError("every attribute word has the same association; "
                         "the effect size divides by zero.")
    num = float(np.mean(s1) - np.mean(s2))
    return RichResult(payload={
        "estimate": num / sd, "numerator": num, "std": sd, "ddof": ddof,
        "s_A1": [float(v) for v in s1], "s_A2": [float(v) for v in s2],
        "n": int(union.size),
        "method": "WEAT effect size (Kamath Eq 6.6)"})


def cheatsheet():
    return "km082: (mean s over A_1 - over A_2) / std over the union"
