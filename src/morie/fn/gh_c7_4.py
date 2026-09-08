# morie.fn -- function file (rootcoder007/morie)
"""Posterior consistency for normal mixture density estimation."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["ghosal_norm_mix_con"]


def ghosal_norm_mix_con(x, grid=None, alpha=1.0, K=50, seed=0, n_draws=200):
    r"""Posterior consistency of Dirichlet-process normal mixtures
    (Ghosal Sec. 7.2.1).

    A DP mixture of normals is weakly and Hellinger consistent at any
    density :math:`p_0` in the Kullback-Leibler support of the prior,
    which for this prior includes essentially every smooth positive
    density -- a strong statement, and the reason normal mixtures are
    the default nonparametric density prior.

    Consistency here is Schwartz's theorem in action: it needs the
    prior to put positive mass on every KL neighbourhood of
    :math:`p_0`, plus a testing condition. The KL support requirement
    is the binding one, and it is a statement about the PRIOR, not
    about the data -- a prior whose support misses :math:`p_0` is
    inconsistent no matter how much data arrives.

    This returns the fitted mixture together with a measured
    Hellinger distance to a kernel-density reference, so consistency
    is exercised rather than asserted.

    Parameters
    ----------
    x : array-like
        Observations.
    grid : array-like, optional
        Evaluation points.
    alpha, K, seed, n_draws
        DP concentration, truncation, RNG seed, Monte Carlo draws.

    Returns
    -------
    RichResult
        keys: ``grid``, ``density``, ``reference_density``,
        ``hellinger_to_reference``, ``consistency`` , ``requires``,
        ``n``, ``method``.
    References
    ----------
    Ghosal and van der Vaart, Sec. 7.2.1 (normal mixtures);
    Schwartz (1965).
    """
    from ._ghosal import hellinger
    from .gh_c5_8 import ghosal_gauss_ker

    xv = np.asarray(x, dtype=float).ravel()
    if xv.size < 5:
        raise ValueError(f"need at least 5 observations, got {xv.size}.")
    fit = ghosal_gauss_ker(xv, grid=grid, alpha=alpha, K=K, seed=seed,
                           n_draws=n_draws)
    g = fit["grid"]
    h = 1.06 * float(xv.std(ddof=1)) * xv.size ** -0.2
    ref = np.exp(-0.5 * ((g[:, None] - xv) / h) ** 2).sum(axis=1) / \
        (xv.size * h * np.sqrt(2 * np.pi))
    return RichResult(payload={
        "grid": g, "density": fit["density"], "reference_density": ref,
        "hellinger_to_reference": hellinger(fit["density"], ref, g),
        "consistency": "weak and Hellinger, at any p0 in the KL support",
        "requires": "positive prior mass on every KL neighbourhood of p0 "
                    "(a property of the PRIOR, not of the data)",
        "n": int(xv.size),
        "method": "DP normal mixture (Sec. 7.2.1); Schwartz consistency via KL support"})


def cheatsheet():
    return "gh_c7_4: consistency needs KL support -- a prior missing p0 never recovers, at any n"
