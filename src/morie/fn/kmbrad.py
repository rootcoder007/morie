# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Ch 5: the Bradley-Terry preference probability."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_bradley_terry_preference"]


def kamath_bradley_terry_preference(r_w, r_l):
    r"""P(y_w > y_l | x) = sigmoid(r(x, y_w) - r(x, y_l)).

    Only the reward DIFFERENCE matters, so the sigmoid is evaluated
    from the difference alone, in the numerically stable branch form
    (no exp of a large positive number). The matching training loss
    -log of this probability is ``morie.fn.alrmt``, which is not
    re-derived here.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 5, Bradley-Terry
    Preference Model; Bradley and Terry (1952).

    Examples
    --------
    >>> import math
    >>> out = kamath_bradley_terry_preference(2.0, 0.0)
    >>> abs(out["estimate"] - 1 / (1 + math.exp(-2.0))) < 1e-15
    True
    >>> kamath_bradley_terry_preference(1.0, 1.0)["estimate"]
    0.5
    """
    w = np.atleast_1d(np.asarray(r_w, dtype=float))
    l = np.atleast_1d(np.asarray(r_l, dtype=float))
    if w.shape != l.shape:
        raise ValueError(
            f"{w.size} winner scores but {l.size} loser scores.")
    if w.size == 0:
        raise ValueError("no preference pairs were given.")
    if not (np.all(np.isfinite(w)) and np.all(np.isfinite(l))):
        raise ValueError("reward scores must be finite.")
    d = w - l
    p = np.where(d >= 0, 1.0 / (1.0 + np.exp(-np.abs(d))),
                 np.exp(-np.abs(d)) / (1.0 + np.exp(-np.abs(d))))
    est = float(p[0]) if p.size == 1 else [float(v) for v in p]
    return RichResult(payload={
        "estimate": est, "p_pref": [float(v) for v in p],
        "reward_difference": [float(v) for v in d], "n": int(p.size),
        "method": "Bradley-Terry preference probability (Kamath Ch 5)"})


def cheatsheet():
    return "kmbrad: sigmoid of the winner-minus-loser reward difference"
