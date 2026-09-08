# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exponential learning-rate decay."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_lr_exponential_schedule"]

_METHOD = "Exponential LR decay"


def geron_lr_exponential_schedule(eta0, gamma, t):
    r"""Multiply the learning rate by a constant factor every step.

    .. math::
        \eta_t = \eta_0\, \gamma^{t}

    Constant *ratio*, not constant difference: the rate falls by the
    same percentage each step, so it approaches zero without ever
    reaching it.  The half-life :math:`\ln 2 / \ln(1/\gamma)` is
    reported because it is the number that actually tells you whether
    the schedule matches the length of your run -- a ``gamma`` that
    looks close to 1 can still kill the rate in a few hundred steps.

    The full curve for ``0 .. t`` is returned, and its monotonicity is
    verified rather than assumed.

    Parameters
    ----------
    eta0 : float
        Initial rate, positive.
    gamma : float
        Decay factor in ``(0, 1]``. ``gamma = 1`` is a constant rate.
    t : int
        Final step, non-negative.

    Returns
    -------
    RichResult
        Payload keys ``eta`` (value at ``t``), ``schedule`` (length
        ``t + 1``), ``half_life``, ``is_monotone_decreasing``,
        ``fraction_remaining``, ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 11, Exponential Learning Rate section.

    Examples
    --------
    Halving each step: after 3 steps an eighth is left.

    >>> r = geron_lr_exponential_schedule(0.1, 0.5, t=3)
    >>> r["eta"]
    0.0125
    >>> r["schedule"]
    [0.1, 0.05, 0.025, 0.0125]
    >>> r["half_life"]
    1.0

    A gentle-looking gamma still decays fast over a long run -- 0.99 has
    a half-life of about 69 steps:

    >>> r2 = geron_lr_exponential_schedule(1.0, 0.99, t=100)
    >>> round(r2["half_life"], 6)
    68.967564
    >>> round(r2["fraction_remaining"], 6)
    0.366032

    ``gamma = 1`` is a flat schedule, and the monotonicity flag stays
    true (non-increasing):

    >>> geron_lr_exponential_schedule(0.1, 1.0, t=5)["schedule"][-1]
    0.1
    """
    eta0 = float(eta0)
    if not np.isfinite(eta0) or eta0 <= 0:
        raise ValueError(f"eta0 must be a positive finite float, got {eta0}.")
    gamma = float(gamma)
    if not (0.0 < gamma <= 1.0):
        raise ValueError(
            f"gamma must lie in (0, 1]; gamma > 1 would grow the learning rate "
            f"without bound and gamma <= 0 is not a decay. Got {gamma}."
        )
    t = int(t)
    if t < 0:
        raise ValueError(f"t must be non-negative, got {t}.")

    steps = np.arange(t + 1)
    curve = eta0 * gamma**steps
    mono = bool(np.all(np.diff(curve) <= 1e-18))
    half = float("inf") if gamma == 1.0 else float(np.log(2.0) / np.log(1.0 / gamma))

    return RichResult(
        title="Exponential LR schedule",
        summary_lines=[("eta(t)", float(curve[t])), ("gamma", gamma),
                       ("Half-life (steps)", half)],
        payload={
            "eta": float(curve[t]),
            "schedule": curve.tolist(),
            "half_life": half,
            "is_monotone_decreasing": mono,
            "fraction_remaining": float(curve[t] / eta0),
            "eta0": eta0,
            "gamma": gamma,
            "t": t,
            "estimate": float(curve[t]),
            "n": int(t + 1),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grlrex: eta_t = eta_0 gamma^t; half-life ln2/ln(1/gamma), full curve returned"
