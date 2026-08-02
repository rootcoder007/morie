# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Smith's 1cycle LR schedule: triangular warm-up then anneal + momentum mirror."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_1cycle_schedule"]

_METHOD = "1cycle learning-rate schedule (Smith 2018)"


def geron_1cycle_schedule(eta_min, eta_max, t, T, mom_max=0.95, mom_min=0.85):
    r"""Build the full 1cycle learning-rate and momentum curves.

    The learning rate rises linearly from ``eta_min`` to ``eta_max`` over
    the first half of the run and falls linearly back to ``eta_min`` over
    the second half.  Momentum mirrors it: high when the learning rate is
    low, low when the learning rate is high, so that the two never fight.

    Concretely, with :math:`h = \lfloor (T-1)/2 \rfloor` the peak step,

    .. math::
        \eta_i = \operatorname{interp}\bigl(i;\ [0, h, T-1],\
                 [\eta_{\min}, \eta_{\max}, \eta_{\min}]\bigr)

    and the momentum uses the same knots with the endpoint values
    swapped.

    Parameters
    ----------
    eta_min, eta_max : float
        Learning-rate bounds. ``eta_min`` must be positive and strictly
        below ``eta_max``.
    t : int
        Step at which to report the headline value, ``0 <= t < T``.
    T : int
        Total number of steps in the cycle, at least 2.
    mom_max, mom_min : float, optional
        Momentum bounds for the mirrored curve.

    Returns
    -------
    RichResult
        Payload keys ``lr_schedule``, ``momentum_schedule``, ``peak_step``,
        ``lr_at_t``, ``momentum_at_t``, ``estimate`` (the learning rate at
        step ``t``), ``n``, ``method``.

    References
    ----------
    Géron Ch 11, 1cycle section (Smith 2018).

    Examples
    --------
    >>> r = geron_1cycle_schedule(0.1, 0.5, t=1, T=5)
    >>> [round(x, 6) for x in r["lr_schedule"]]
    [0.1, 0.3, 0.5, 0.3, 0.1]
    >>> [round(x, 6) for x in r["momentum_schedule"]]
    [0.95, 0.9, 0.85, 0.9, 0.95]
    >>> round(float(r), 6)
    0.3
    """
    eta_min = float(eta_min)
    eta_max = float(eta_max)
    if not np.isfinite(eta_min) or not np.isfinite(eta_max):
        raise ValueError("eta_min and eta_max must be finite.")
    if eta_min <= 0:
        raise ValueError(f"eta_min must be positive, got {eta_min}.")
    if eta_max <= eta_min:
        raise ValueError(f"eta_max ({eta_max}) must exceed eta_min ({eta_min}).")
    T = int(T)
    if T < 2:
        raise ValueError(f"T must be at least 2 steps, got {T}.")
    t = int(t)
    if not (0 <= t < T):
        raise ValueError(f"t must satisfy 0 <= t < T={T}, got {t}.")
    mom_max = float(mom_max)
    mom_min = float(mom_min)
    if not (0.0 <= mom_min < mom_max < 1.0):
        raise ValueError(
            f"momentum bounds must satisfy 0 <= mom_min < mom_max < 1, "
            f"got mom_min={mom_min}, mom_max={mom_max}."
        )

    steps = np.arange(T, dtype=float)
    peak = (T - 1) // 2
    knots = np.array([0.0, float(peak), float(T - 1)])
    # np.interp needs strictly increasing x; T=2 collapses peak onto 0.
    if peak == 0:
        knots = np.array([0.0, float(T - 1)])
        lr = np.interp(steps, knots, [eta_min, eta_max])
        mom = np.interp(steps, knots, [mom_max, mom_min])
    else:
        lr = np.interp(steps, knots, [eta_min, eta_max, eta_min])
        mom = np.interp(steps, knots, [mom_max, mom_min, mom_max])

    return RichResult(
        title="1cycle schedule",
        summary_lines=[("Peak step", peak), ("LR at t", float(lr[t]))],
        payload={
            "lr_schedule": lr.tolist(),
            "momentum_schedule": mom.tolist(),
            "peak_step": int(peak),
            "lr_at_t": float(lr[t]),
            "momentum_at_t": float(mom[t]),
            "eta_min": eta_min,
            "eta_max": eta_max,
            "estimate": float(lr[t]),
            "n": T,
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "gr1cy: Smith's 1cycle LR schedule -- triangular warm-up then anneal, momentum mirrored"
