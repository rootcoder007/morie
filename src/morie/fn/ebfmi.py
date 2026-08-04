# morie.fn -- function file (rootcoder007/morie)
"""Energy Bayesian fraction of missing information."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['ebfmi', 'energy_bayesian_fmi']


def ebfmi(energy):
    """Energy Bayesian fraction of missing information.

    Hamiltonian Monte Carlo explores a level set at fixed energy and moves between level sets only through momentum resampling. When the energy the resampling injects is small against the spread of the marginal energy distribution, the chain crawls across level sets no matter how good the trajectories look, and this ratio is what exposes it. Note the asymmetric index ranges -- N successive differences over N+1 squared deviations -- which is how the estimator is printed in the source and is kept here rather than tidied into matching counts. No pass/fail threshold is returned: the commonly quoted cut-off is not in the source and inventing one would be worse than reporting the number.


    Formula: EBFMI = sum_{n=1}^{N} (E_n - E_{n-1})^2 / sum_{n=0}^{N} (E_n - Ebar)^2

    Parameters
    ----------
    energy : array-like
        Energies per iteration; a list of lists is treated as one chain per row.

    Returns
    -------
    RichResult
        ``ebfmi`` (per chain), ``min_ebfmi``, ``n_chains``, ``n``.

    References
    ----------
    Betancourt (2016), Diagnosing Suboptimal Cotangent Disintegrations
    in Hamiltonian Monte Carlo, arXiv:1604.00695.  Verified against the
    paper: the estimator is the displayed equation for BFMI-hat.
    """
    E = energy
    if not (isinstance(E, (list, tuple)) and E and isinstance(E[0], (list, tuple))):
        E = [C.vec(E)]
    else:
        E = [C.vec(row) for row in E]
    out = []
    for e in E:
        N = len(e)
        if N < 2:
            raise ValueError("need at least two energies per chain")
        num = sum((e[n] - e[n - 1]) ** 2 for n in range(1, N))
        mu = sum(e) / N
        den = sum((v - mu) ** 2 for v in e)
        out.append(num / den if den > 0 else float("nan"))
    return RichResult(payload={
        "ebfmi": out, "min_ebfmi": min(out), "n_chains": len(E),
        "n": len(E[0]), "method": "Energy Bayesian fraction of missing information"})


energy_bayesian_fmi = ebfmi


def cheatsheet():
    return "ebfmi: Energy Bayesian fraction of missing information."
