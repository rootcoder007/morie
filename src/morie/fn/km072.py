# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 5.8: the Bradley-Terry preference probability."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_ch5_bradley_terry_pref"]


def _sigmoid(z):
    """Stable logistic: no exp overflow at either tail."""
    z = float(z)
    if z >= 0:
        return float(1.0 / (1.0 + np.exp(-z)))
    e = np.exp(z)
    return float(e / (1.0 + e))


def kamath_ch5_bradley_terry_pref(r_star, y_w, y_l):
    """p*(y_w > y_l | x) = exp(r*(x,y_w)) / (exp(r*(x,y_w)) +
    exp(r*(x,y_l))).

    ``r_star`` is a mapping response -> reward or a callable. Written
    as a ratio of exponentials it overflows for large rewards; dividing
    through by exp(r_w) gives the identical sigmoid of the MARGIN,
    which is what is computed and what km073 states as Eq 5.9.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 5, Eq 5.8, printed
    p. 209.

    Examples
    --------
    >>> out = kamath_ch5_bradley_terry_pref({"a": 1.0, "b": 0.0}, "a", "b")
    >>> round(out["estimate"], 10)
    0.7310585786
    >>> round(kamath_ch5_bradley_terry_pref({"a": 5.0, "b": 5.0},
    ...                                     "a", "b")["estimate"], 12)
    0.5
    """
    if callable(r_star):
        rw, rl = float(r_star(y_w)), float(r_star(y_l))
    elif isinstance(r_star, dict):
        for k in (y_w, y_l):
            if k not in r_star:
                raise ValueError(f"response {k!r} has no reward in r_star.")
        rw, rl = float(r_star[y_w]), float(r_star[y_l])
    else:
        raise ValueError("r_star must be a mapping response -> reward or a "
                         "callable.")
    if not (np.isfinite(rw) and np.isfinite(rl)):
        raise ValueError("a reward is not finite.")
    p = _sigmoid(rw - rl)
    return RichResult(payload={
        "estimate": p, "margin": rw - rl, "r_w": rw, "r_l": rl,
        "p_reversed": 1.0 - p, "n": 2,
        "method": "Bradley-Terry preference probability (Kamath Eq 5.8)"})


def cheatsheet():
    return "km072: p(y_w > y_l) = exp(r_w)/(exp(r_w)+exp(r_l))"
