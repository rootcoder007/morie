# morie.fn -- function file (rootcoder007/morie)
"""Bulk effective sample size."""

from __future__ import annotations

from . import _array_core as np

from ._mcmc import ess_from_chains, rank_normalize
from ._richresult import RichResult

__all__ = ["effective_sample_size_bulk"]


def effective_sample_size_bulk(chains):
    r"""Bulk-ESS: the effective sample size for the centre of the distribution.

    Computed on **rank-normalised** draws, which is what makes it robust to
    heavy tails and to distributions with no finite variance -- the plain ESS
    is undefined for a Cauchy posterior, while bulk-ESS is not.

    Bulk-ESS governs the reliability of central summaries: the posterior mean,
    the median, the bulk of the density. It says nothing about the tails,
    which is why it is always reported alongside
    :func:`~morie.fn.esstl.effective_sample_size_tail` rather than instead of
    it. Vehtari et al. recommend at least 100 per chain before any summary is
    trusted.

    Parameters
    ----------
    chains : array-like
        Draws ``(m, n)`` or ``(n,)``.

    Returns
    -------
    RichResult
        ``ess_bulk``, ``n_draws``, ``efficiency``, ``sufficient``.

    References
    ----------
    Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Burkner, P.-C.
        (2021). Rank-normalization, folding, and localization: An improved
        R-hat for assessing convergence of MCMC. *Bayesian Analysis*,
        16(2), 667-718.

    Examples
    --------
    Independent draws give bulk-ESS near the draw count.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> r = effective_sample_size_bulk(rng.normal(size=(4, 1000)))
    >>> bool(r["efficiency"] > 0.85)
    True

    Rank normalisation makes it well defined for a distribution with no
    variance at all, where the plain ESS is not.

    >>> heavy = rng.standard_cauchy(size=(4, 1000))
    >>> bool(np.isfinite(effective_sample_size_bulk(heavy)["ess_bulk"]))
    True

    The sufficiency flag applies the 100-per-chain rule.

    >>> bool(r["sufficient"])
    True
    >>> bool(not effective_sample_size_bulk(rng.normal(size=(2, 10)))["sufficient"])
    True
    """
    C = np.atleast_2d(np.asarray(chains, dtype=float))
    m, n = C.shape
    if n < 4:
        raise ValueError("need at least 4 draws per chain")
    ess = ess_from_chains(rank_normalize(C))
    total = m * n
    return RichResult(
        title="Bulk effective sample size",
        summary_lines=[("draws", int(total)), ("bulk ESS", float(ess)),
                       ("efficiency", float(ess / total))],
        warnings=([f"bulk ESS below 100 per chain ({ess:.0f} for {m} chains); "
                   "central summaries are unreliable"] if ess < 100 * m else []),
        payload={
            "ess_bulk": float(ess), "n_draws": int(total),
            "efficiency": float(ess / total),
            "sufficient": bool(ess >= 100 * m),
            "n_chains": int(m), "method": "effective_sample_size_bulk",
        },
    )


def cheatsheet():
    return "essbk: rank-normalised so it survives heavy tails; governs CENTRAL summaries only"
