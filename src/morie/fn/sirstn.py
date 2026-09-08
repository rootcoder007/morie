# morie.fn -- wave2 slice x_4_01 (rootcoder007/morie)
"""Stochastic SIR by Gillespie's direct method."""

from __future__ import annotations

import math

from . import _tail1core as C
from ._richresult import RichResult

__all__ = ["sir_stochastic"]


def sir_stochastic(S0, I0, beta, gamma, T, seed=1):
    r"""Simulate the SIR continuous-time Markov chain exactly, using
    Gillespie's direct method (the "stochastic simulation algorithm").

    Two reaction channels act on integer counts :math:`(S, I, R)`:

    .. math::

        a_1 = \beta\,S\,I / N \;\; (S,I) \to (S-1, I+1), \qquad
        a_2 = \gamma\,I \;\; (I,R) \to (I-1, R+1),

    with total propensity :math:`a_0 = a_1 + a_2`.  Given two independent
    uniforms :math:`u_1, u_2` on (0,1) the direct method draws the waiting
    time and the channel as

    .. math::

        \tau = -\ln(u_1)/a_0, \qquad
        \text{channel } 1 \text{ iff } u_2 a_0 < a_1 ,

    which is exact -- no time discretisation is involved.  The chain stops
    when :math:`I = 0` (the absorbing state) or when the clock passes *T*.

    The uniform stream is the Lehmer minstd generator
    :math:`s \leftarrow 48271\,s \bmod (2^{31}-1)` shared with every other
    arm of this package, so a given ``seed`` reproduces the same trajectory
    in Python and in R bit for bit.

    Parameters
    ----------
    S0, I0 : int
        Initial susceptible and infectious counts.  N = S0 + I0.
    beta : float
        Transmission rate.
    gamma : float
        Recovery rate.
    T : float
        Time horizon.
    seed : int, default 1
        Seed for the shared minstd stream.

    Returns
    -------
    RichResult
        ``estimate`` is the final size, the number recovered when the chain
        stops.

    References
    ----------
    Gillespie, D. T. (1977). Exact stochastic simulation of coupled chemical
    reactions. Journal of Physical Chemistry 81(25), 2340-2361.
    doi:10.1021/j100540a008
    """
    S = int(S0)
    I = int(I0)
    R = 0
    beta = float(beta)
    gamma = float(gamma)
    T = float(T)
    if S < 0 or I < 0:
        raise ValueError("sir_stochastic: S0 and I0 must be non-negative")
    if beta < 0.0 or gamma < 0.0:
        raise ValueError("sir_stochastic: beta and gamma must be non-negative")
    if T < 0.0:
        raise ValueError("sir_stochastic: T must be non-negative")
    N = S + I
    if N <= 0:
        raise ValueError("sir_stochastic: total population must be positive")

    rng = C.Lcg(seed)
    t = 0.0
    peak_I = float(I)
    peak_time = 0.0
    n_events = 0
    n_infections = 0
    extinction_time = float("nan")

    while True:
        a1 = beta * S * I / N
        a2 = gamma * I
        a0 = a1 + a2
        if a0 <= 0.0:
            extinction_time = t
            break
        tau = -math.log(rng.unif()) / a0
        if t + tau > T:
            t = T
            break
        t = t + tau
        if rng.unif() * a0 < a1:
            S -= 1
            I += 1
            n_infections += 1
            if I > peak_I:
                peak_I = float(I)
                peak_time = t
        else:
            I -= 1
            R += 1
        n_events += 1

    r0 = beta / gamma if gamma > 0.0 else float("inf")
    return RichResult(
        payload={
            "estimate": float(R),
            "S": float(S),
            "I": float(I),
            "R": float(R),
            "N": float(N),
            "final_size": float(R),
            "attack_rate": float(R) / N,
            "peak_I": peak_I,
            "peak_time": peak_time,
            "t_end": t,
            "extinction_time": extinction_time,
            "n_events": float(n_events),
            "n_infections": float(n_infections),
            "R0": r0,
            "beta": beta,
            "gamma": gamma,
            "T": T,
            "seed": float(int(seed)),
            "method": "Stochastic SIR, Gillespie direct method (Gillespie 1977)",
        }
    )


def cheatsheet():
    return "sirstn: stochastic SIR by Gillespie's direct method"


# compact alias per ledger/NAMING.md
sirstochastic = sir_stochastic
