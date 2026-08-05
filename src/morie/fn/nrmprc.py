# morie.fn -- function file (rootcoder007/morie)
"""Normalized inverse-Gaussian process.

DUPLICATE: the NIG Levy mass is already implemented in ``gh_c14_14``
(public name ``ghosal_nig_proc``).  Per ledger/wave2/DUPMAP.tsv the
quadrature is not repeated here; this module delegates to it and adds
only the data-side summary its own signature asks for.
"""

from .gh_c14_14 import ghosal_nig_proc as _nig
from ._richresult import RichResult

__all__ = ["normalized_inverse_gauss"]


def normalized_inverse_gauss(y, alpha=1.0, tau=1.0, u_max=10.0, n_grid=6000):
    """Normalized inverse-Gaussian random probability measure.

    The NIG process normalizes a completely random measure whose Levy
    intensity is the tilted stable-1/2 density

        rho(u) = (2 pi)^{-1/2} u^{-3/2} exp(-tau^2 u / 2),

    so ``int u rho(u) du = 1 / tau`` exactly.  Its tails are heavier
    than the Dirichlet process's, which is the reason it is used in its
    place; the total-mass quadrature is delegated to
    :func:`morie.fn.gh_c14_14.ghosal_nig_proc`.

    Parameters
    ----------
    y : array-like
        Observed values; only their count and their number of distinct
        values (the realized partition) enter the summary.
    alpha : float, default 1.0
        Total mass of the base measure.
    tau : float, default 1.0
        Exponential tilting parameter of the Levy density.
    u_max, n_grid : float, int
        Quadrature range and midpoint-rule grid passed through.

    Returns
    -------
    RichResult
        ``estimate`` (quadrature total jump mass), ``theory``
        (``1 / tau``), ``gap``, ``alpha``, ``tau``, ``n``, ``n_distinct``.

    References
    ----------
    Lijoi, A., Mena, R. H. & Prunster, I. (2005).  Hierarchical mixture
    modeling with normalized inverse-Gaussian priors.  Journal of the
    American Statistical Association, 100(472), 1278--1291.
    Ghosal, S. & van der Vaart, A. (2017).  Fundamentals of
    Nonparametric Bayesian Inference, CUP, section 14.6.
    """
    vals = [float(v) for v in (y if hasattr(y, "__len__") else [y])]
    t = float(tau)
    if t <= 0.0:
        raise ValueError("normalized_inverse_gauss: tau must be positive")
    if float(alpha) <= 0.0:
        raise ValueError("normalized_inverse_gauss: alpha must be positive")
    base = _nig(alpha_par=t, u_max=float(u_max), n_grid=int(n_grid))
    seen = []
    for v in vals:
        if v not in seen:
            seen.append(v)
    return RichResult(payload={
        "estimate": float(base["estimate"]),
        "theory": float(base["theory"]),
        "gap": float(base["gap"]),
        "alpha": float(alpha), "tau": t,
        "n": len(vals), "n_distinct": len(seen),
        "method": "Normalized inverse-Gaussian process (Lijoi-Mena-Prunster 2005)"})


def cheatsheet():
    return "nrmprc: Normalized inverse-Gaussian process (delegates to gh_c14_14)"


normalizedinversegauss = normalized_inverse_gauss
