# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 3.11: the T5 template generation objective (LM-BFF)."""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_ch3_t5_template_obj"]


def _apply(T, x_in, y):
    if callable(T):
        return T(x_in, y)
    if isinstance(T, str):
        return T.format(x=x_in, y=y)
    raise ValueError("T must be a callable (x_in, y) -> text or a format "
                     "string using {x} and {y}.")


def kamath_ch3_t5_template_obj(D_train, T, T5):
    """sum_{(x_in,y) in D_train} log P_T5(T | T(x_in, y)).

    The quantity maximised when searching for a template: ``T5`` is a
    callable (T, filled_input) -> probability of emitting the template,
    validated to lie in (0, 1]. A probability of 0 makes the objective
    -inf, which is the mathematics; it is raised as an error rather
    than returned silently because a template that can never be
    generated cannot be ranked.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 3, Eq 3.11, printed
    p. 108.

    Examples
    --------
    >>> import math
    >>> D = [("great movie", "pos"), ("awful movie", "neg")]
    >>> out = kamath_ch3_t5_template_obj(D, "{x} It was {y}",
    ...                                  lambda T, s: 0.5)
    >>> abs(out["estimate"] - 2 * math.log(0.5)) < 1e-12
    True
    """
    pairs = list(D_train)
    if not pairs:
        raise ValueError("D_train is empty; a sum over no examples is "
                         "undefined, not 0.")
    if not callable(T5):
        raise ValueError("T5 must be a callable (T, filled_input) -> "
                         "probability.")
    logs, filled = [], []
    for x_in, y in pairs:
        s = _apply(T, x_in, y)
        p = float(T5(T, s))
        if not (0.0 < p <= 1.0):
            raise ValueError(
                f"P_T5 returned {p:.6g} for {s!r}; it must lie in (0, 1].")
        filled.append(s)
        logs.append(math.log(p))
    arr = np.asarray(logs, dtype=float)
    return RichResult(payload={
        "estimate": float(arr.sum()), "per_example": [float(v) for v in arr],
        "filled_inputs": filled, "n": len(pairs),
        "method": "T5 template generation objective (Kamath Eq 3.11)"})


def cheatsheet():
    return "km052: sum of log P_T5(T | T(x_in, y)) over D_train"
