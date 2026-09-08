# morie.fn -- slice s03 (rootcoder007/morie)
"""Learning-rate schedule for the AlphaZero value/policy network.

Source consulted (FETCHED): Silver, D. et al. (2018), arXiv:1712.01815:
"The learning rate was set to 0.2 for each game, and was dropped three
times (to 0.02, 0.002 and 0.0002 respectively) during the course of
training."  AlphaZero therefore used a *step* schedule, not a cosine
one -- that is what the paper prints, and it is available here as
``kind="step"`` with exactly those four rates.

The module's own formula line asks for the cosine schedule of Loshchilov
and Hutter (2017), SGDR: stochastic gradient descent with warm restarts,
arXiv:1608.03983,

    lr_t = lr_0 * 0.5 (1 + cos(pi t / T))

which is ``kind="cosine"``, the default.  The distinction is kept
explicit rather than papered over: the cosine curve is not AlphaZero's
own schedule.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401

from ._richresult import RichResult

__all__ = ["alphazero_value_lr"]

# The four rates the AlphaZero paper prints, with the fractions of
# training at which the drops are usually placed.
_STEPS = (0.2, 0.02, 0.002, 0.0002)


def alphazero_value_lr(t, T, lr_0=0.2, kind="cosine", floor=0.0):
    """Learning rate at step t of a T-step run.

    Parameters
    ----------
    t : float
        Current step.
    T : float
        Total number of steps.
    lr_0 : float
        Initial learning rate; AlphaZero uses 0.2.
    kind : {"cosine", "step"}
        Cosine annealing, or AlphaZero's own four-step drop.
    floor : float
        Lower bound applied to the cosine curve.

    Returns
    -------
    RichResult with payload:
        estimate : lr_t
        lr, frac, kind
    """
    tt = float(t)
    TT = float(T)
    frac = tt / TT if TT > 0.0 else 0.0
    if frac < 0.0:
        frac = 0.0
    if frac > 1.0:
        frac = 1.0
    if kind == "step":
        idx = int(frac * 4.0)
        if idx > 3:
            idx = 3
        lr = float(lr_0) * (_STEPS[idx] / _STEPS[0])
    else:
        lr = float(floor) + (float(lr_0) - float(floor)) * 0.5 * (
            1.0 + math.cos(math.pi * frac)
        )
    return RichResult(
        title="AlphaZero learning-rate schedule",
        summary_lines=[("lr", lr), ("kind", kind)],
        payload={
            "estimate": lr,
            "lr": lr,
            "frac": frac,
            "kind": kind,
            "method": "Cosine annealing (default) or AlphaZero's printed step schedule",
        },
    )


def cheatsheet():
    return "agvslr: AlphaZero learning-rate schedule"
