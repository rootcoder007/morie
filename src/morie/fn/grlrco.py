# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cosine annealing learning-rate schedule."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_lr_cosine_annealing"]

_METHOD = "Cosine annealing LR schedule"


def geron_lr_cosine_annealing(eta_min, eta_max, t, T):
    r"""Anneal the learning rate along a half cosine.

    .. math::
        \eta_t = \eta_{\min} + \tfrac12(\eta_{\max}-\eta_{\min})
        \Bigl(1 + \cos\frac{\pi t}{T}\Bigr)

    The cosine is flat at both ends and steepest in the middle: the rate
    lingers high early (where large steps pay), drops fast through the
    middle, and settles gently into ``eta_min`` instead of falling off a
    cliff the way a step schedule does.

    The whole curve for ``t = 0 .. T`` is returned, not just the value
    at ``t`` -- a schedule is only checkable as a curve, and
    ``is_monotone_decreasing`` is verified on it here.

    Parameters
    ----------
    eta_min, eta_max : float
        Bounds, ``eta_min <= eta_max``, both non-negative.
    t : int
        Step to report, in ``0 .. T``.
    T : int
        Length of the annealing cycle, at least 1.

    Returns
    -------
    RichResult
        Payload keys ``eta`` (value at ``t``), ``schedule`` (length
        ``T + 1``), ``eta_min``, ``eta_max``, ``halfway_value``,
        ``is_monotone_decreasing``, ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 11, Cosine Annealing section (Loshchilov and Hutter 2017).

    Examples
    --------
    The endpoints are exact and the midpoint is the arithmetic mean of
    the two -- ``cos(pi/2) = 0``:

    >>> r = geron_lr_cosine_annealing(0.0, 1.0, t=5, T=10)
    >>> r["eta"]
    0.5
    >>> r["schedule"][0], round(r["schedule"][-1], 12)
    (1.0, 0.0)
    >>> r["is_monotone_decreasing"]
    True

    A quarter of the way in, only ``1 - cos(pi/4)`` of the range is
    gone -- the curve is deliberately flat early:

    >>> round(geron_lr_cosine_annealing(0.0, 1.0, t=2, T=8)["eta"], 10)
    0.8535533906
    """
    eta_min = float(eta_min)
    eta_max = float(eta_max)
    if not np.isfinite(eta_min) or not np.isfinite(eta_max):
        raise ValueError("eta_min and eta_max must be finite.")
    if eta_min < 0 or eta_max < 0:
        raise ValueError(f"learning rates must be non-negative, got {eta_min}, {eta_max}.")
    if eta_min > eta_max:
        raise ValueError(f"eta_min ({eta_min}) must not exceed eta_max ({eta_max}).")
    T = int(T)
    if T < 1:
        raise ValueError(f"T must be at least 1, got {T}; T = 0 divides by zero.")
    t = int(t)
    if not (0 <= t <= T):
        raise ValueError(f"t must lie in [0, {T}], got {t}.")

    steps = np.arange(T + 1)
    curve = eta_min + 0.5 * (eta_max - eta_min) * (1.0 + np.cos(np.pi * steps / T))
    mono = bool(np.all(np.diff(curve) <= 1e-15))

    return RichResult(
        title="Cosine annealing schedule",
        summary_lines=[("eta(t)", float(curve[t])), ("t / T", f"{t} / {T}")],
        payload={
            "eta": float(curve[t]),
            "schedule": curve.tolist(),
            "eta_min": eta_min,
            "eta_max": eta_max,
            "halfway_value": float(eta_min + 0.5 * (eta_max - eta_min)),
            "is_monotone_decreasing": mono,
            "t": t,
            "T": T,
            "estimate": float(curve[t]),
            "n": int(T + 1),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grlrco: eta_t = eta_min + 0.5(eta_max-eta_min)(1+cos(pi t/T)); full curve returned"
