# morie.fn -- function file (rootcoder007/morie)
"""Restricted mean survival time."""

from __future__ import annotations

from . import _array_core as np

from ._containers import DescriptiveResult


def rmst_estimate(time, event, *, tau: float | None = None) -> DescriptiveResult:
    """RMST: area under Kaplan-Meier curve up to time tau.

    Parameters
    ----------
    time : array-like
        Observed survival times.
    event : array-like
        Event indicator (1 = event, 0 = censored).
    tau : float, optional
        Restriction time. Default: max observed time.

    Returns
    -------
    DescriptiveResult
    """
    from morie.fn.km import kaplan_meier

    km_result = kaplan_meier(time, event)
    t = km_result.times
    s = km_result.survival
    if tau is None:
        tau = float(t[-1])
    # The Kaplan-Meier curve is a STEP function: it is constant on
    # [t_i, t_{i+1}) and drops at t_{i+1}.  Integrating it with the
    # trapezoid rule linearly interpolates across each step and therefore
    # understates the area by half of every drop times its width -- on
    # time = [1, 2, 3], event = all, tau = 3 it returned 1.5 where the
    # closed form is exactly 2.  Summed rectangles are the definition.
    area = 0.0
    prev_t = 0.0
    prev_s = 1.0
    for i in range(len(t)):
        ti = float(t[i])
        if ti >= tau:
            break
        area += prev_s * (ti - prev_t)
        prev_t = ti
        prev_s = float(s[i])
    area += prev_s * (float(tau) - prev_t)
    area = float(area)
    return DescriptiveResult(
        name="RMST",
        value=area,
        extra={"tau": tau, "n_events": km_result.n_events},
    )


rmst = rmst_estimate


def cheatsheet() -> str:
    return "rmst_estimate({}) -> Restricted mean survival time."


# compact alias per ledger/NAMING.md
rmstestimate = rmst_estimate
