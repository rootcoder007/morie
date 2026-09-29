# morie.fn -- function file from book-equation translation pipeline (rootcoder007/morie)
"""Bayesian Fraction of Missing Information (BFMI)."""

from __future__ import annotations

__all__ = ["bayesian_fmi", "bfmi"]

from typing import Any


def bayesian_fmi(
    energy,
) -> dict[str, Any]:
    """
    Bayesian Fraction of Missing Information (BFMI / E-BFMI).

    Diagnostic for HMC / NUTS samplers: how well the momentum resampling
    explores the energy distribution. Betancourt's estimator compares the
    energy change between successive iterations with the marginal spread
    of the energy,

    .. math::

        \\widehat{\text{E-BFMI}} = \frac{\\sum_{n=1}^{N-1} (E_n - E_{n-1})^2}
                                   {\\sum_{n=0}^{N-1} (E_n - \bar E)^2}.

    Values below 0.3 indicate that the sampler has difficulty exploring
    the target distribution. (The ratio of the *variances* of the
    differences and of the energies -- the form this module used to
    compute -- centres the differences and changes both divisors, and is
    not Betancourt's estimator.)

    Parameters
    ----------
    energy : array-like
        Hamiltonian energy values from each HMC/NUTS iteration (n,).

    Returns
    -------
    dict
        bfmi : float
        adequate : bool (bfmi >= 0.3)
        energy_var : float (sample variance of the energy, divisor n - 1)
        transition_var : float (mean squared energy transition)

    References
    ----------
    Betancourt, M. (2016). Diagnosing suboptimal cotangent disintegrations
    in Hamiltonian Monte Carlo. arXiv:1604.00695.
    Betancourt, M. (2017). A conceptual introduction to Hamiltonian
    Monte Carlo. arXiv:1701.02434.

    Examples
    --------
    >>> round(bayesian_fmi([1.0, 3.0, 2.0, 5.0, 4.0])["bfmi"], 12)
    1.5
    """
    e = [float(v) for v in (energy.tolist() if hasattr(energy, "tolist") else energy)]
    n = len(e)
    if n < 3:
        raise ValueError("Need at least 3 energy values.")
    m = 0.0
    for v in e:
        m += v
    m /= n
    ss = 0.0
    for v in e:
        ss += (v - m) ** 2
    sd2 = 0.0
    for i in range(1, n):
        sd2 += (e[i] - e[i - 1]) ** 2
    bfmi_val = 1.0 if ss < 1e-30 else sd2 / ss
    return {
        "bfmi": bfmi_val,
        "adequate": bool(bfmi_val >= 0.3),
        "energy_var": ss / (n - 1),
        "transition_var": sd2 / (n - 1),
    }


bfmi = bayesian_fmi


def cheatsheet() -> str:
    return "bayesian_fmi(energy) -> Bayesian Fraction of Missing Information."


# compact alias per ledger/NAMING.md
bayesianfmi = bayesian_fmi
