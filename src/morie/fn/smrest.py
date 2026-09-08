# morie.fn -- function file (rootcoder007/morie)
"""Standardised mortality ratio, indirect standardisation."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['smrind', 'standardized_mortality_ratio']


def smrind(observed, expected, alpha=0.05):
    """Standardised mortality ratio, indirect standardisation.

    Indirect standardisation applies a reference population's age-specific rates to the study population's age structure to get E, then compares the observed count to it. The interval is the exact Poisson one on the count, not a normal approximation, which matters precisely in the small-O case where SMRs are usually reported. This module delegates to morie.fn.smr rather than keeping a second copy of the same arithmetic.


    Formula: SMR = O / E; exact Poisson limits chi2_{alpha/2, 2O}/2 and chi2_{1-alpha/2, 2(O+1)}/2, divided by E

    Parameters
    ----------
    observed : int
        Observed deaths.
    expected : float
        Expected deaths under the reference rates.
    alpha : float
        Two-sided significance level.

    Returns
    -------
    RichResult
        ``smr``, ``ci_lower``, ``ci_upper``, ``observed``, ``expected``.

    References
    ----------
    Breslow and Day (1987), Statistical Methods in Cancer Research
    Volume II: The Design and Analysis of Cohort Studies, IARC.  Not held
    locally; SMR = O/E with exact Poisson limits on O is the standard
    published form, and is what the existing morie.fn.smr implements.
    """
    from .smr import standardized_mortality_ratio as _smr
    out = _smr(observed, expected, alpha=alpha)
    return RichResult(payload={
        "smr": out["smr"], "ci_lower": out["ci_lower"],
        "ci_upper": out["ci_upper"], "observed": float(observed),
        "expected": float(expected),
        "method": "Standardised mortality ratio (indirect)"})


standardized_mortality_ratio = smrind


def cheatsheet():
    return "smrest: Standardised mortality ratio, indirect standardisation."
