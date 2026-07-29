# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 5.10: the DPO preference BEFORE Z(x) is cancelled."""

import math

import numpy as np

from ._richresult import RichResult
from .km073 import kamath_ch5_pref_sigmoid_form
from .km075 import _implicit_rewards, kamath_ch5_dpo_pref_simplified

__all__ = ["kamath_ch5_dpo_pref_substituted"]


def kamath_ch5_dpo_pref_substituted(pi_star, pi_ref, beta, Z=None):
    """p* = sigma(beta log[pi*(y_w)/pi_ref(y_w)] + beta log Z
    - beta log[pi*(y_l)/pi_ref(y_l)] - beta log Z).

    Eq 5.7 substituted into Eq 5.9 with the two beta log Z(x) terms
    STILL PRESENT. They are carried explicitly here and shown to
    cancel: the result is compared against km075's Z-free form and any
    disagreement is an error, not a rounding excuse. That cancellation
    is the whole reason DPO needs no partition function.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 5, Eq 5.10, printed
    p. 210.

    Examples
    --------
    >>> out = kamath_ch5_dpo_pref_substituted([0.75, 0.25], [0.5, 0.5],
    ...                                       1.0, Z=1000.0)
    >>> round(out["estimate"], 12), out["z_terms_cancel"]
    (0.75, True)
    """
    rw, rl, beta = _implicit_rewards(pi_star, pi_ref, beta)
    Zv = 1.0 if Z is None else float(Z)
    if Zv <= 0:
        raise ValueError("Z must be strictly positive.")
    off = beta * math.log(Zv)
    inner = kamath_ch5_pref_sigmoid_form([rw + off, rl + off])
    simple = kamath_ch5_dpo_pref_simplified(pi_star, pi_ref, beta)
    cancels = abs(inner["estimate"] - simple["estimate"]) < 1e-12
    if not cancels:
        raise ValueError(
            "the beta log Z terms failed to cancel; substituted "
            f"{inner['estimate']!r} vs simplified {simple['estimate']!r}.")
    return RichResult(payload={
        "estimate": inner["estimate"], "margin": inner["margin"],
        "z_offset": float(off), "z_terms_cancel": bool(cancels),
        "simplified": simple["estimate"], "beta": beta, "Z": Zv, "n": 2,
        "method": "DPO preference with Z carried explicitly "
                  "(Kamath Eq 5.10)"})


def cheatsheet():
    return "km074: Eq 5.9 + Eq 5.7, the beta log Z terms cancel"
