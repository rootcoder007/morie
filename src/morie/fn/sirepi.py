# morie.fn -- function file (rootcoder007/morie)
"""SIR compartmental model."""

from __future__ import annotations

import math

from ._richresult import RichResult

__all__ = ["sir_compartmental"]


def _rhs(s, i, beta, gamma, N):
    inf = beta * s * i / N
    return -inf, inf - gamma * i, gamma * i


def sir_compartmental(S0, I0, R0, beta, gamma, T, dt=0.1):
    r"""SIR compartmental model.

    Kermack and McKendrick's (1927) closed-population model ``dS/dt =
    -beta S I / N``, ``dI/dt = beta S I / N - gamma I``, ``dR/dt = gamma I``,
    ``N = S + I + R``, integrated on ``[0, T]`` by the classical fourth-order
    Runge-Kutta method with step ``dt`` (the last step shortened to land on
    ``T``). Also returned: the basic reproduction number ``beta / gamma``,
    the epidemic peak, and the final size ``S_inf`` solving ``S_inf = S0
    exp(-(beta / (gamma N)) (N - S_inf - R0))`` (from ``dS/dR = -beta S /
    (gamma N)``; Brauer, van den Driessche and Wu 2008, ch. 2).

    Parameters
    ----------
    S0, I0, R0 : float
        Initial susceptible, infected and removed counts.
    beta : float
        Transmission rate.
    gamma : float
        Removal rate.
    T : float
        Time horizon.
    dt : float
        Runge-Kutta step.

    Returns
    -------
    RichResult
        ``t``, ``S``, ``I``, ``R`` (grids), ``R0_basic``, ``peak_I``,
        ``peak_time``, ``final_S`` (asymptotic), ``attack_rate`` (``1 -
        final_S / S0``), ``estimate`` (= attack rate).

    References
    ----------
    Kermack, W. O. and McKendrick, A. G. (1927). A contribution to the mathematical theory of
    epidemics. *Proceedings of the Royal Society A*, 115(772), 700-721.

    Brauer, F., van den Driessche, P. and Wu, J. (eds) (2008). *Mathematical Epidemiology*. Springer.

    Examples
    --------
    >>> r = sir_compartmental(990, 10, 0, 0.3, 0.1, 50, dt=0.5)
    >>> round(r["I"][-1], 8), round(r["final_S"], 8)
    (88.11232825, 58.7973648)
    """
    s, i, rr = float(S0), float(I0), float(R0)
    beta, gamma, T, dt = float(beta), float(gamma), float(T), float(dt)
    if min(s, i, rr) < 0 or beta < 0 or gamma <= 0 or T <= 0 or dt <= 0:
        raise ValueError("need non-negative counts, beta >= 0, gamma, T, dt > 0")
    N = s + i + rr
    t = 0.0
    ts, S, Inf, R = [0.0], [s], [i], [rr]
    while t < T - 1e-12:
        h = min(dt, T - t)
        k1 = _rhs(s, i, beta, gamma, N)
        k2 = _rhs(s + h / 2 * k1[0], i + h / 2 * k1[1], beta, gamma, N)
        k3 = _rhs(s + h / 2 * k2[0], i + h / 2 * k2[1], beta, gamma, N)
        k4 = _rhs(s + h * k3[0], i + h * k3[1], beta, gamma, N)
        s += h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        i += h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        rr += h / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        t += h
        ts.append(t)
        S.append(s)
        Inf.append(i)
        R.append(rr)
    k = max(range(len(Inf)), key=lambda j: Inf[j])
    # final size: root of g(x) = x - S0 exp(-(beta/(gamma N)) (N - x - R0)) on (0, S0]
    a = beta / (gamma * N)
    lo, hi = 0.0, float(S0)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid - float(S0) * math.exp(-a * (N - mid - float(R0))) < 0:
            lo = mid
        else:
            hi = mid
    fin = 0.5 * (lo + hi)
    attack = 1.0 - fin / float(S0) if S0 > 0 else 0.0
    return RichResult(
        payload={
            "t": ts,
            "S": S,
            "I": Inf,
            "R": R,
            "R0_basic": beta / gamma,
            "peak_I": Inf[k],
            "peak_time": ts[k],
            "final_S": fin,
            "attack_rate": attack,
            "estimate": attack,
            "method": "SIR by classical Runge-Kutta; final size by bisection",
        }
    )


def cheatsheet():
    return "sirepi: SIR model by RK4 with R0, peak and final size"
