# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""1cycle policy: warm up then anneal the learning rate."""

import numpy as np

from ._richresult import RichResult
from .gr1cy import geron_1cycle_schedule

__all__ = ["geron_one_cycle"]

_METHOD = "1cycle learning-rate policy"


def geron_one_cycle(t, T, lr_max, lr_min, mom_max=0.95, mom_min=0.85):
    """
    1cycle policy: warm up then anneal the learning rate.

    Formula: lr ramps up to lr_max then down to lr_min over T steps

    The schedule itself is delegated to
    :func:`morie.fn.gr1cy.geron_1cycle_schedule`, which already
    implements the triangular ramp and the mirrored momentum ramp; this
    entry supplies the ``(t, T, lr_max, lr_min)`` argument order of the
    Géron listing and adds the phase bookkeeping -- which half of the
    cycle step ``t`` falls in, and the peak step.

    Momentum moves opposite to the learning rate: high momentum while
    the rate is small, low momentum at the peak, so the effective step
    size stays under control at the top of the ramp.

    Parameters
    ----------
    t : int
        Step to report, ``0 <= t < T`` -- the same 0-based indexing the
        delegate uses, so ``lr_schedule[t]`` is the reported rate.
    T : int
        Total steps in the cycle (at least 2).
    lr_max : float
        Peak learning rate.
    lr_min : float
        Learning rate at both ends of the cycle.
    mom_max, mom_min : float
        Momentum at the ends and at the peak respectively.

    Returns
    -------
    result : RichResult
        Keys: lr, momentum, lr_schedule, momentum_schedule, peak_step,
        phase, estimate, n, method.

    Examples
    --------
    A five-step cycle from 0.1 to 0.5 and back, peaking at step 3:

    >>> r = geron_one_cycle(t=0, T=5, lr_max=0.5, lr_min=0.1)
    >>> [round(x, 6) for x in r["lr_schedule"]]
    [0.1, 0.3, 0.5, 0.3, 0.1]
    >>> r["peak_step"], r["phase"]
    (2, 'warmup')
    >>> round(float(r["lr"]), 6)
    0.1

    At the peak the momentum is at its minimum:

    >>> p = geron_one_cycle(t=2, T=5, lr_max=0.5, lr_min=0.1)
    >>> round(float(p["lr"]), 6), round(float(p["momentum"]), 6), p["phase"]
    (0.5, 0.85, 'peak')

    >>> geron_one_cycle(t=4, T=5, lr_max=0.5, lr_min=0.1)["phase"]
    'anneal'
    >>> round(float(geron_one_cycle(t=4, T=5, lr_max=0.5, lr_min=0.1)["momentum"]), 6)
    0.95

    References
    ----------
    Géron Ch 11
    """
    T_int = int(T)
    t_int = int(t)
    if T_int < 2:
        raise ValueError(f"geron_one_cycle: T must be at least 2 steps, got {T!r}")
    if not (0 <= t_int < T_int):
        raise ValueError(f"geron_one_cycle: t must lie in 0..{T_int - 1}, got {t!r}")
    hi, lo = float(lr_max), float(lr_min)
    if not np.isfinite(hi) or not np.isfinite(lo):
        raise ValueError("geron_one_cycle: lr_max and lr_min must be finite")
    if lo <= 0:
        raise ValueError(f"geron_one_cycle: lr_min must be positive, got {lo}")
    if hi <= lo:
        raise ValueError(f"geron_one_cycle: lr_max ({hi}) must exceed lr_min ({lo}) for a cycle to exist")

    inner = geron_1cycle_schedule(lo, hi, t=t_int, T=T_int, mom_max=float(mom_max), mom_min=float(mom_min))
    peak = int(inner["peak_step"])
    phase = "peak" if t_int == peak else ("warmup" if t_int < peak else "anneal")

    return RichResult(
        title="1cycle policy",
        summary_lines=[
            ("Step", f"{t_int} of {T_int - 1}"),
            ("Learning rate", float(inner["lr_schedule"][t_int])),
            ("Momentum", float(inner["momentum_schedule"][t_int])),
            ("Phase", phase),
        ],
        interpretation=(
            "Momentum is the mirror image of the learning rate: it dips exactly where the rate peaks."
        ),
        payload={
            "lr": float(inner["lr_schedule"][t_int]),
            "momentum": float(inner["momentum_schedule"][t_int]),
            "lr_schedule": inner["lr_schedule"],
            "momentum_schedule": inner["momentum_schedule"],
            "peak_step": peak,
            "phase": phase,
            "estimate": float(inner["lr_schedule"][t_int]),
            "n": T_int,
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hml1c: 1cycle LR policy (delegates the ramp to gr1cy) with phase and mirrored momentum"
