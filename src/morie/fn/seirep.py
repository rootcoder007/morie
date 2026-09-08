# morie.fn -- wave2 slice x_4_01 (rootcoder007/morie)
"""SEIR compartmental model with an exposed (latent) class."""

from __future__ import annotations

from ._richresult import RichResult

__all__ = ["seir_compartmental"]


def seir_compartmental(S, E, I, R, beta, sigma, gamma, t_max=160.0, dt=0.1):
    r"""Integrate the deterministic SEIR model.

    .. math::

        \frac{dS}{dt} = -\beta S I / N, \qquad
        \frac{dE}{dt} = \beta S I / N - \sigma E, \\
        \frac{dI}{dt} = \sigma E - \gamma I, \qquad
        \frac{dR}{dt} = \gamma I .

    The exposed class E holds individuals who are infected but not yet
    infectious; they progress at rate :math:`\sigma`, so the mean latent
    period is :math:`1/\sigma`.  The basic reproduction number is
    :math:`R_0 = \beta/\gamma`: the latent stage delays but does not alter
    the number of secondary cases, because every exposed individual
    eventually becomes infectious in this model.  Population
    :math:`N = S + E + I + R` is conserved.

    Integrated with classical fourth-order Runge-Kutta at a fixed step, so
    the result is deterministic and identical in every language arm.

    Parameters
    ----------
    S, E, I, R : float
        Initial counts in the four compartments.
    beta : float
        Transmission rate.
    sigma : float
        Rate of progression from exposed to infectious (1 / latent period).
    gamma : float
        Recovery rate (1 / infectious period).
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
    Hethcote, H. W. (2000). The mathematics of infectious diseases.
    SIAM Review 42(4), 599-653. doi:10.1137/S0036144500371907
    """
    y = [float(S), float(E), float(I), float(R)]
    beta = float(beta)
    sigma = float(sigma)
    gamma = float(gamma)
    t_max = float(t_max)
    dt = float(dt)
    if any(v < 0.0 for v in y):
        raise ValueError("seir_compartmental: compartment sizes must be non-negative")
    if beta < 0.0 or sigma < 0.0 or gamma < 0.0:
        raise ValueError("seir_compartmental: rates must be non-negative")
    if dt <= 0.0 or t_max < 0.0:
        raise ValueError("seir_compartmental: need dt > 0 and t_max >= 0")
    N = y[0] + y[1] + y[2] + y[3]
    if N <= 0.0:
        raise ValueError("seir_compartmental: total population must be positive")

    def deriv(v):
        f = beta * v[0] * v[2] / N
        return [-f, f - sigma * v[1], sigma * v[1] - gamma * v[2], gamma * v[2]]

    nsteps = int(round(t_max / dt))
    peak_I = y[2]
    peak_time = 0.0
    for step in range(nsteps):
        k1 = deriv(y)
        k2 = deriv([y[i] + 0.5 * dt * k1[i] for i in range(4)])
        k3 = deriv([y[i] + 0.5 * dt * k2[i] for i in range(4)])
        k4 = deriv([y[i] + dt * k3[i] for i in range(4)])
        y = [y[i] + (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) for i in range(4)]
        if y[2] > peak_I:
            peak_I = y[2]
            peak_time = (step + 1) * dt

    r0 = beta / gamma if gamma > 0.0 else float("inf")
    return RichResult(
        payload={
            "estimate": y[3],
            "S": y[0],
            "E": y[1],
            "I": y[2],
            "R": y[3],
            "N": N,
            "R0": r0,
            "latent_period": (1.0 / sigma) if sigma > 0.0 else float("inf"),
            "infectious_period": (1.0 / gamma) if gamma > 0.0 else float("inf"),
            "peak_I": peak_I,
            "peak_time": peak_time,
            "final_size": y[3],
            "conservation_error": abs(y[0] + y[1] + y[2] + y[3] - N),
            "beta": beta,
            "sigma": sigma,
            "gamma": gamma,
            "t_max": t_max,
            "dt": dt,
            "method": "SEIR compartmental model (Hethcote 2000)",
        }
    )


def cheatsheet():
    return "seirep: SEIR compartmental model with exposed class (Hethcote 2000)"


# compact alias per ledger/NAMING.md
seircompartmental = seir_compartmental
