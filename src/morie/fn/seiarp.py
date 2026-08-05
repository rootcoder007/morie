# morie.fn -- wave2 slice x_4_01 (rootcoder007/morie)
"""SEIRA: SEIR with a parallel asymptomatic infectious compartment."""

from __future__ import annotations

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["seira_asymptomatic"]


def seira_asymptomatic(S, E, I, A, R, params, t_max=160.0, dt=0.1):
    r"""Integrate an SEIRA model: the latent class splits into two
    infectious streams, symptomatic and asymptomatic.

    .. math::

        \frac{dS}{dt} = -S\,(\beta I + \kappa\beta A)/N, \qquad
        \frac{dE}{dt} = S\,(\beta I + \kappa\beta A)/N - \sigma E, \\
        \frac{dI}{dt} = p\,\sigma E - \gamma I, \qquad
        \frac{dA}{dt} = (1-p)\,\sigma E - \gamma_a A, \qquad
        \frac{dR}{dt} = \gamma I + \gamma_a A .

    A fraction *p* of those leaving the latent class become symptomatic and
    a fraction :math:`1-p` asymptomatic; asymptomatic carriers transmit at a
    relative infectiousness :math:`\kappa` and clear at their own rate
    :math:`\gamma_a`.  Because the two infectious streams are in parallel,
    Anderson & May's "average over the infectious classes" rule gives

    .. math::

        R_0 = \beta\left[\frac{p}{\gamma}
                       + \frac{\kappa\,(1-p)}{\gamma_a}\right],

    the sum over routes of (transmission rate) x (probability of entering the
    route) x (mean time spent in it).  Setting :math:`p = 1` collapses the A
    compartment and recovers the SEIR value :math:`R_0 = \beta/\gamma`.

    Parameters
    ----------
    S, E, I, A, R : float
        Initial counts in the five compartments.
    params : array-like
        Six rates in this order: ``beta``, ``sigma``, ``gamma``, ``p``,
        ``kappa``, ``gamma_a``.
    t_max : float, default 160.0
        Integration horizon.
    dt : float, default 0.1
        Runge-Kutta step.

    Returns
    -------
    RichResult
        ``estimate`` is the final size, R at ``t_max``.

    References
    ----------
    Anderson, R. M. & May, R. M. (1991). Infectious Diseases of Humans:
    Dynamics and Control. Oxford University Press. ISBN 0-19-854040-X.
    """
    pr = [float(v) for v in np.atleast_1d(np.asarray(params, dtype=float)).tolist()]
    if len(pr) != 6:
        raise ValueError(
            "seira_asymptomatic: params must be (beta, sigma, gamma, p, kappa, gamma_a)"
        )
    beta, sigma, gamma, p, kappa, gamma_a = pr
    if beta < 0.0 or sigma < 0.0 or gamma < 0.0 or kappa < 0.0 or gamma_a < 0.0:
        raise ValueError("seira_asymptomatic: rates must be non-negative")
    if p < 0.0 or p > 1.0:
        raise ValueError("seira_asymptomatic: p must lie in [0, 1]")
    y = [float(S), float(E), float(I), float(A), float(R)]
    if any(v < 0.0 for v in y):
        raise ValueError("seira_asymptomatic: compartment sizes must be non-negative")
    t_max = float(t_max)
    dt = float(dt)
    if dt <= 0.0 or t_max < 0.0:
        raise ValueError("seira_asymptomatic: need dt > 0 and t_max >= 0")
    N = y[0] + y[1] + y[2] + y[3] + y[4]
    if N <= 0.0:
        raise ValueError("seira_asymptomatic: total population must be positive")

    def deriv(v):
        f = v[0] * (beta * v[2] + kappa * beta * v[3]) / N
        return [
            -f,
            f - sigma * v[1],
            p * sigma * v[1] - gamma * v[2],
            (1.0 - p) * sigma * v[1] - gamma_a * v[3],
            gamma * v[2] + gamma_a * v[3],
        ]

    nsteps = int(round(t_max / dt))
    peak_I = y[2]
    peak_time = 0.0
    for step in range(nsteps):
        k1 = deriv(y)
        k2 = deriv([y[i] + 0.5 * dt * k1[i] for i in range(5)])
        k3 = deriv([y[i] + 0.5 * dt * k2[i] for i in range(5)])
        k4 = deriv([y[i] + dt * k3[i] for i in range(5)])
        y = [y[i] + (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) for i in range(5)]
        if y[2] > peak_I:
            peak_I = y[2]
            peak_time = (step + 1) * dt

    sym = (p / gamma) if gamma > 0.0 else float("inf")
    asym = (kappa * (1.0 - p) / gamma_a) if gamma_a > 0.0 else float("inf")
    r0 = beta * (sym + asym)

    return RichResult(
        payload={
            "estimate": y[4],
            "S": y[0],
            "E": y[1],
            "I": y[2],
            "A": y[3],
            "R": y[4],
            "N": N,
            "R0": r0,
            "R0_symptomatic": beta * sym,
            "R0_asymptomatic": beta * asym,
            "asymptomatic_fraction": 1.0 - p,
            "peak_I": peak_I,
            "peak_time": peak_time,
            "final_size": y[4],
            "conservation_error": abs(y[0] + y[1] + y[2] + y[3] + y[4] - N),
            "beta": beta,
            "sigma": sigma,
            "gamma": gamma,
            "p": p,
            "kappa": kappa,
            "gamma_a": gamma_a,
            "t_max": t_max,
            "dt": dt,
            "method": "SEIRA with asymptomatic compartment (Anderson & May 1991)",
        }
    )


def cheatsheet():
    return "seiarp: SEIRA, SEIR with an asymptomatic infectious class"


# compact alias per ledger/NAMING.md
seiraasymptomatic = seira_asymptomatic
