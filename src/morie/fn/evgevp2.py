# morie.fn -- function file (rootcoder007/morie)
"""Probability-weighted-moment estimator for the GEV -- alias."""

from ._richresult import RichResult

__all__ = ["ev_gev_pwm", "evt_gev_pwm"]


def ev_gev_pwm(block_maxima):
    """The PWM estimator of Hosking, Wallis and Wood (1985) for the
    GEV. Since Hosking's (1990) L-moments are exactly the linear
    combinations l1 = b0, l2 = 2 b1 - b0, l3 = 6 b2 - 6 b1 + b0 of
    the probability-weighted moments, the PWM fit and the L-moment
    fit are THE SAME ESTIMATOR in different coordinates. One
    implementation therefore serves both catalogue entries: the
    computation is :func:`morie.fn.evgevlm.ev_gev_lmoments`, and the
    PWMs themselves are added to the output for readers of the 1985
    paper.

    References
    ----------
    Hosking, J. R. M., Wallis, J. R. and Wood, E. F. (1985),
    "Estimation of the generalized extreme-value distribution by the
    method of probability-weighted moments", *Technometrics*
    27:251-261.
    """
    from ._evt import pwm_b
    from .evgevlm import ev_gev_lmoments

    out = ev_gev_lmoments(block_maxima)
    payload = dict(out)
    payload["b0"] = pwm_b(block_maxima, 0)
    payload["b1"] = pwm_b(block_maxima, 1)
    payload["b2"] = pwm_b(block_maxima, 2)
    payload["alias_of"] = "morie.fn.evgevlm.ev_gev_lmoments"
    payload["same_estimator_because"] = (
        "L-moments are linear combinations of the PWMs, so the two fits "
        "coincide exactly")
    return RichResult(payload=payload)


def cheatsheet():
    return "evgevp2: PWM fit == L-moment fit, one implementation"


#: Catalogue alias for :func:`ev_gev_pwm`.
evt_gev_pwm = ev_gev_pwm


# compact alias per ledger/NAMING.md
evgevpwm = ev_gev_pwm


# compact alias per ledger/NAMING.md
evtgevpwm = evt_gev_pwm
