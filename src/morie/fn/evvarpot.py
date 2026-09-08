# morie.fn -- function file (rootcoder007/morie)
"""Value-at-risk from a peaks-over-threshold GPD tail."""

import math

from ._richresult import RichResult

__all__ = ["evt_pot_var"]


def evt_pot_var(u, sigma, xi, zeta_u, p):
    """
    Value-at-risk from a fitted GPD tail

    Formula: VaR_p = u + (sigma/xi) (((1-p)/zeta_u)^(-xi) - 1)

    Inverting the tail estimate P(X > x) = zeta_u (1 + xi (x-u)/sigma)^(-1/xi).
    The xi = 0 limit is u + sigma log(zeta_u/(1-p)), used when |xi| is
    below 1e-12 so the formula stays continuous.

    Parameters
    ----------
    u : float
        Threshold.
    sigma : float
        GPD scale, strictly positive.
    xi : float
        GPD shape.
    zeta_u : float
        Exceedance rate P(X > u), in (0, 1].
    p : float
        VaR level, in (0, 1); p must exceed 1 - zeta_u.

    Returns
    -------
    result : dict
        Keys: VaR, estimate, tail_prob, p.

    References
    ----------
    McNeil & Frey (2000), J. Empirical Finance 7(3-4):271-300.
    """
    u = float(u)
    sigma = float(sigma)
    xi = float(xi)
    zeta_u = float(zeta_u)
    p = float(p)
    if not (sigma > 0.0):
        raise ValueError("sigma must be strictly positive")
    if not (0.0 < zeta_u <= 1.0):
        raise ValueError("zeta_u must lie in (0, 1]")
    if not (0.0 < p < 1.0):
        raise ValueError("p must lie strictly in (0, 1)")
    if 1.0 - p > zeta_u:
        raise ValueError("p is below the threshold exceedance rate; "
                         "the GPD tail says nothing there")
    r = (1.0 - p) / zeta_u
    if abs(xi) < 1e-12:
        var = u + sigma * math.log(1.0 / r)
    else:
        var = u + (sigma / xi) * (r ** (-xi) - 1.0)
    # the tail probability implied by the answer, as a self-consistency read
    z = 1.0 + xi * (var - u) / sigma
    if abs(xi) < 1e-12:
        tail = zeta_u * math.exp(-(var - u) / sigma)
    else:
        tail = zeta_u * z ** (-1.0 / xi) if z > 0.0 else 0.0
    return RichResult(payload={
        "VaR": var,
        "estimate": var,
        "tail_prob": tail,
        "p": p,
        "method": "GPD value-at-risk above a POT threshold",
    })


def cheatsheet():
    return "evvarpot: GPD value-at-risk above a POT threshold"


# compact alias per ledger/NAMING.md
evtpotvar = evt_pot_var
