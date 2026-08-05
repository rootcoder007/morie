# morie.fn -- function file (rootcoder007/morie)
"""Expected shortfall from a peaks-over-threshold GPD tail."""

from ._richresult import RichResult

__all__ = ["evt_pot_es"]


def evt_pot_es(u, sigma, xi, VaR):
    """
    Expected shortfall of a GPD tail above a threshold

    Formula: ES_p = (VaR_p + sigma - xi u) / (1 - xi)

    E[X | X > VaR] for the generalised Pareto tail.  The mean excess of
    a GPD is finite only for xi < 1, so xi >= 1 is refused rather than
    returned as a number.  At xi = 0 the expression collapses to
    VaR + sigma, the memoryless exponential mean excess.

    Parameters
    ----------
    u : float
        Threshold the GPD was fitted above.
    sigma : float
        GPD scale at that threshold, strictly positive.
    xi : float
        GPD shape.
    VaR : float
        Value-at-risk at the level of interest.

    Returns
    -------
    result : dict
        Keys: ES, estimate, ratio, xi.

    References
    ----------
    McNeil & Frey (2000), J. Empirical Finance 7(3-4):271-300.
    """
    u = float(u)
    sigma = float(sigma)
    xi = float(xi)
    VaR = float(VaR)
    if not (sigma > 0.0):
        raise ValueError("sigma must be strictly positive")
    if xi >= 1.0:
        raise ValueError("expected shortfall is infinite for xi >= 1")
    es = (VaR + sigma - xi * u) / (1.0 - xi)
    return RichResult(payload={
        "ES": es,
        "estimate": es,
        "ratio": es / VaR if VaR != 0.0 else float("nan"),
        "xi": xi,
        "method": "GPD expected shortfall above a POT threshold",
    })


def cheatsheet():
    return "evespot: GPD expected shortfall above a POT threshold"


# compact alias per ledger/NAMING.md
evtpotes = evt_pot_es
