# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 4.4: the LoRA objective, optimised over Theta alone."""

from . import _array_core as np

from ._richresult import RichResult
from .km056 import _sequence_objective

__all__ = ["kamath_ch4_lora_obj"]


def kamath_ch4_lora_obj(Theta, Phi_0, x, y):
    """max_Theta sum_{(x,y)} sum_t log p_{Phi_0 + dPhi(Theta)}(y_t|x,y_<t).

    The SAME sum as Eq 4.3 (km056, delegated) over a different
    parameter set: ``Theta`` is the adapted model Phi_0 + dPhi(Theta)
    as a callable (x_i, y_prefix, y_t) -> probability, ``Phi_0`` the
    frozen base model in the same form. The base objective is scored
    too, so the caller sees what the adapter bought -- the whole claim
    of LoRA is that |Theta| << |Phi_0| loses nothing.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 4, Eq 4.4, printed
    p. 151.

    Examples
    --------
    >>> import math
    >>> out = kamath_ch4_lora_obj(lambda xi, p, t: 0.5,
    ...                           lambda xi, p, t: 0.25, ["doc"], [["a"]])
    >>> abs(out["estimate"] - math.log(0.5)) < 1e-12
    True
    >>> abs(out["improvement"] - math.log(2.0)) < 1e-12
    True
    """
    adapted, per_pair = _sequence_objective(Theta, x, y)
    base, base_pairs = _sequence_objective(Phi_0, x, y)
    return RichResult(payload={
        "estimate": float(adapted), "base_objective": float(base),
        "improvement": float(adapted - base), "per_pair": per_pair,
        "base_per_pair": base_pairs, "n": len(per_pair),
        "method": "LoRA objective over Theta (Kamath Eq 4.4)"})


def cheatsheet():
    return "km057: Eq 4.3's sum scored under Phi_0 + dPhi(Theta)"
