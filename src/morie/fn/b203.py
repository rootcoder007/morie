# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Burkov's Eq 2.3: the shorthand notations for Eq 2.2 agree."""

from . import _array_core as np

from ._richresult import RichResult
from .b202 import burkov_lm_ch2_lm_next_token

__all__ = ["burkov_lm_ch2_lm_shorthand"]


def burkov_lm_ch2_lm_shorthand(t_next, s):
    """Pr(t_{L+1} | t_1..t_L) and Pr(t_{L+1} | s) are the same number.

    Eq 2.3 states a notational equivalence; the operational check is
    that evaluating through both spellings gives the identical value,
    which delegates to the Eq 2.2 estimator and reports agreement.

    References: Burkov LM (2025), Ch 2, Eq 2.3, p. 76.

    Examples
    --------
    >>> burkov_lm_ch2_lm_shorthand("b", ["a", "b", "a"])["notations_agree"]
    True
    """
    seq = [str(t) for t in np.atleast_1d(np.asarray(s, dtype=object))]
    via_seq = burkov_lm_ch2_lm_next_token(t_next, seq)["estimate"]
    via_tokens = burkov_lm_ch2_lm_next_token(t_next, list(seq))["estimate"]
    return RichResult(payload={
        "estimate": float(via_seq), "via_sequence": float(via_seq),
        "via_tokens": float(via_tokens),
        "notations_agree": via_seq == via_tokens, "n": len(seq),
        "method": "Shorthand equivalence for Eq 2.2 (Burkov Eq 2.3)"})


def cheatsheet():
    return "b203: next-token shorthand equivalence (Burkov Eq 2.3)"
