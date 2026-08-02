# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 5.1: the pairwise reward-model loss."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_ch5_reward_loss_pairwise"]


def _bt_loss(margins):
    """-mean log sigmoid(margin), computed as mean softplus(-margin).

    The stable form: log(1 + exp(-m)) never overflows for large |m|
    where sigmoid would round to 0 or 1 and log would blow up.
    km067 and km076 import this -- all three equations are one loss.
    """
    m = np.atleast_1d(np.asarray(margins, dtype=float))
    if m.size == 0:
        raise ValueError("no preference pairs; an expectation over an "
                         "empty dataset is undefined, not 0.")
    if not np.all(np.isfinite(m)):
        raise ValueError("a reward margin is not finite.")
    per = np.logaddexp(0.0, -m)
    return float(per.mean()), per


def kamath_ch5_reward_loss_pairwise(r_theta, x, y_0, y_1, i):
    """loss(r) = -E[log sigma(r(x, y_i) - r(x, y_{1-i}))].

    ``i`` names which of the two candidates the human preferred, so the
    margin is always (preferred - rejected). ``r_theta`` is the reward
    model as a callable (x, y) -> scalar. The loss depends only on the
    margin, never on the reward scale -- adding a constant to every
    reward leaves it unchanged, which the tests assert.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 5, Eq 5.1, printed
    p. 197.

    Examples
    --------
    >>> import math
    >>> r = lambda x, y: 1.0 if y == "good" else 0.0
    >>> out = kamath_ch5_reward_loss_pairwise(
    ...     r, ["p"], ["good"], ["bad"], [0])
    >>> abs(out["estimate"] - math.log(1 + math.exp(-1))) < 1e-12
    True
    """
    if not callable(r_theta):
        raise ValueError("r_theta must be a callable (x, y) -> reward.")
    xs, a, b = list(x), list(y_0), list(y_1)
    ch = [int(v) for v in np.atleast_1d(np.asarray(i)).ravel()]
    if not (len(xs) == len(a) == len(b) == len(ch)):
        raise ValueError(
            f"x, y_0, y_1 and i must have equal length; got {len(xs)}, "
            f"{len(a)}, {len(b)}, {len(ch)}.")
    if any(v not in (0, 1) for v in ch):
        raise ValueError("every entry of i must be 0 or 1.")
    margins = []
    for xi, y0, y1, k in zip(xs, a, b, ch):
        chosen, rejected = (y0, y1) if k == 0 else (y1, y0)
        margins.append(float(r_theta(xi, chosen)) -
                       float(r_theta(xi, rejected)))
    loss, per = _bt_loss(margins)
    return RichResult(payload={
        "estimate": loss, "margins": [float(v) for v in margins],
        "per_pair": [float(v) for v in per], "n": len(xs),
        "method": "pairwise reward-model loss (Kamath Eq 5.1)"})


def cheatsheet():
    return "km065: -mean log sigmoid(r_chosen - r_rejected)"
